"""
Compute features and train kmeans jointly in a single script. This requires
enough of a rewrite that I don't just want to do this by refactoring learn-kmeans.py.

Pros of doing it this way:
    - does not require features to be saved
    - makes it practical to train on far more data
Cons of doing it this way:
    - features need to be recomputed
"""

import argparse
import sys

from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import torch
from transformers import AutoFeatureExtractor
import joblib

from spire.labeler import Featurizer
from spire.data import build_dataloader
from spire.cli import ssl_parser, dataset_parser, randomness_parser


class EarlyStopper:
    """Patience-based early stopper for clustering validation metrics."""

    def __init__(self, patience, is_lower_better):
        self.patience = patience
        self.is_lower_better = is_lower_better
        self.best = None
        self.bad_steps = 0

    def step(self, value):
        """Returns True when training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = value < self.best if self.is_lower_better else value > self.best
        if improved:
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.bad_steps >= self.patience


def extract_validation_features(featurizer, loader, input_field_name, dtype, device):
    """Extract and concatenate all features from a validation loader into a single array."""
    all_features = []
    with torch.no_grad():
        for batch in loader:
            inp = batch[input_field_name].to(dtype=dtype, device=device)
            mask = batch.attention_mask
            if device == "cuda":
                mask = mask.cuda()
            features = featurizer(
                batch=inp,
                attention_mask=mask,
                flatten=True,
                return_pad_percent=True
            )[0].cpu().float().numpy()
            all_features.append(features)
    return np.concatenate(all_features, axis=0)


def compute_val_metrics(kmeans, val_features):
    """Compute inertia, Davies-Bouldin, and Calinski-Harabasz on a held-out feature set."""
    inertia = -kmeans.score(val_features)
    labels = kmeans.predict(val_features)
    db = davies_bouldin_score(val_features, labels)
    ch = calinski_harabasz_score(val_features, labels)
    return {"inertia": inertia, "davies-bouldin": db, "calinski-harabasz": ch}


def main(args):
    if len(args.config) > 1:
        assert args.dataset_weights is None or len(args.config) == len(args.dataset_weights)

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        init=args.init,
        batch_size=args.kmeans_batch_size,
        verbose=args.verbose,
        compute_labels=False,
        init_size=None,
        n_init=args.n_init if args.n_init > 0 else "auto",
        reassignment_ratio=args.reassignment_ratio,
        random_state=args.seed
    )

    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    device = "cpu" if args.cpu else "cuda"

    torch_random = torch.Generator(device="cpu")  # DataLoader always uses CPU
    torch_random.manual_seed(args.torch_seed)

    featurizer = Featurizer(
        args.ssl_model,
        layer=args.layer,
        keep_final_layer_norm=args.keep_final_layer_norm,
        dtype=dtype,
        pooling_width=args.pooling_width,
        pooling_type=args.pooling_type
    )

    featurizer = featurizer.to(device=device)
    featurizer.eval()
    if args.compile:
        featurizer = torch.compile(featurizer)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    loader, _ = build_dataloader(
        config=args.config,
        dataset_weights=args.dataset_weights,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        resample_to=args.resample_to,
        token_batching=args.token_batching,
        example_lengths=args.example_lengths,
        torch_random=torch_random
    )

    val_features = None
    if args.val_config:
        if args.val_dataset_weights is not None:
            assert len(args.val_config) == len(args.val_dataset_weights)
        print("Extracting validation features...")
        val_loader, _ = build_dataloader(
            config=args.val_config,
            dataset_weights=args.val_dataset_weights,
            feature_extractor=feature_extractor,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_examples=args.val_n_examples,
            resample_to=args.resample_to,
        )
        input_field_name = feature_extractor.model_input_names[0]
        val_features = extract_validation_features(featurizer, val_loader, input_field_name, dtype, device)
        print(f"Validation set: {val_features.shape[0]} frames, {val_features.shape[1]} dims")

    _METRIC_LOWER_IS_BETTER = {
        "inertia": True,
        "davies-bouldin": True,
        "calinski-harabasz": False,
    }
    early_stopper = None
    if args.early_stopping_metric in _METRIC_LOWER_IS_BETTER:
        early_stopper = EarlyStopper(
            patience=args.early_stopping_patience,
            is_lower_better=_METRIC_LOWER_IS_BETTER[args.early_stopping_metric],
        )

    n_hours = 0.
    old_centers = None
    feature_buffer = []
    seconds_buffer = []
    n_updates = 0
    frames_seen = 0

    input_field_name = feature_extractor.model_input_names[0]

    with torch.no_grad():
        with tqdm(total=args.max_hours) as pbar:
            for batch in loader:
                inp = batch[input_field_name].to(dtype=dtype, device=device)

                mask = batch.attention_mask
                if device == "cuda":
                    mask = mask.cuda()

                features = featurizer(
                    batch=inp,
                    attention_mask=mask,
                    flatten=True,
                    return_pad_percent=True
                )[0].cpu().float().numpy()

                feature_buffer.append(features)
                frames_seen += features.shape[0]
                seconds_buffer.extend(batch["seconds"])

                if frames_seen >= args.kmeans_batch_size:
                    n_updates += 1
                    kmeans.partial_fit(np.vstack(feature_buffer[:args.kmeans_batch_size]))
                    feature_buffer = []
                    frames_seen = 0

                    batch_hours = sum(seconds_buffer[:args.kmeans_batch_size]) / 3600
                    seconds_buffer = []

                    n_hours += batch_hours  # update hours seen

                    pbar.update(batch_hours)

                    if n_updates % args.validation_steps == 0:
                        stop_training = False

                        if old_centers is not None:
                            delta = np.linalg.norm(kmeans.cluster_centers_ - old_centers) / np.linalg.norm(old_centers)
                            print(f"Centroid delta at step {n_updates}: {delta:.6f}")
                            if args.early_stopping_metric == "centroid-delta" and delta < args.centroid_delta:
                                print(f"Stopping: centroid delta below threshold ({delta} < {args.centroid_delta})")
                                stop_training = True
                        old_centers = kmeans.cluster_centers_.copy()

                        if val_features is not None:
                            metrics = compute_val_metrics(kmeans, val_features)
                            print(
                                f"Validation at step {n_updates}: "
                                f"inertia={metrics['inertia']:.4f}  "
                                f"davies-bouldin={metrics['davies-bouldin']:.4f}  "
                                f"calinski-harabasz={metrics['calinski-harabasz']:.4f}"
                            )
                            if early_stopper is not None and early_stopper.step(metrics[args.early_stopping_metric]):
                                print(
                                    f"Stopping: no improvement in {args.early_stopping_metric} "
                                    f"for {args.early_stopping_patience} consecutive validation steps "
                                    f"(best={early_stopper.best:.6f})"
                                )
                                stop_training = True

                        if args.save_intermediate:
                            joblib.dump(kmeans, args.out_path + "." + str(n_updates))

                        sys.stdout.flush()
                        if stop_training:
                            break

                    if n_hours >= args.max_hours:
                        break
    joblib.dump(kmeans, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[ssl_parser, dataset_parser, randomness_parser])
    parser.add_argument("--max-hours", type=float, default=1000.,
                        help="""If specified, number of hours to train
                             clustering on (otherwise uses whole dataset)""")
    parser.add_argument("--kmeans-batch-size", type=int, default=20000)
    parser.add_argument("--feature-dir")
    parser.add_argument("--hours-per-shard", type=float, default=1.)
    parser.add_argument("--out-path", default="kmeans.joblib")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--init", default="k-means++", choices=["k-means++", "random"])
    parser.add_argument("--n-init", default=0, type=int)
    parser.add_argument("--reassignment-ratio", default=0.01, type=float)  # resetting default to avoid dead centers
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--centroid-delta", type=float, default=1e-4)
    parser.add_argument("--val-config", nargs="*", default=None,
                        help="Dataset config(s) for the held-out validation set (same format as --config)")
    parser.add_argument("--val-dataset-weights", nargs="*", type=float, default=None,
                        help="Per-config sampling weights for the validation set (must match length of --val-config)")
    parser.add_argument("--val-n-examples", type=int, default=0,
                        help="Max number of examples to use from the validation set (0 = all)")
    parser.add_argument("--early-stopping-metric",
                        choices=["centroid-delta", "inertia", "davies-bouldin", "calinski-harabasz"],
                        default="centroid-delta",
                        help="Metric used to trigger early stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="For dataset-based metrics: consecutive validation steps without improvement before stopping")
    args = parser.parse_args()
    main(args)
