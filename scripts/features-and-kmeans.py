"""
Compute features and train kmeans jointly in a single script. This requires
enough of a rewrite that I don't just want to do this by refactoring learn-kmeans.py.

Pros of doing it this way:
    - does not require features to be saved
    - makes it practical to train on far more data
Cons of doing it this way:
    - features need to be recomputed

Note that all of the pros and cons are essentially different ways of framing the
same fact.
"""

import argparse
import sys

from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
from transformers import AutoFeatureExtractor
import joblib

from spire.labeler import Featurizer
from spire.data import build_dataloader
from spire.cli import ssl_parser, dataset_parser, randomness_parser


def main(args):
    if len(args.config) > 1:
        assert args.dataset_weights is None or len(args.config) == len(args.dataset_weights)

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.kmeans_batch_size,
        verbose=args.verbose,
        compute_labels=False,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
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

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)

    featurizer = Featurizer(
        args.ssl_model,
        layer=args.layer,
        no_final_layer_norm=args.no_final_layer_norm,
        dtype=dtype,
        pooling_width=args.pooling_width,
        pooling_type=args.pooling_type
    )

    featurizer = featurizer.to(device=device)
    featurizer.eval()
    if args.compile:
        featurizer = torch.compile(featurizer)

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
        example_lengths=args.example_lengths
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

                # reminder: featurizer is essentially the forward of a HubertModel,
                # but with some layers cut off (is there a more elegant way to
                # do this inside HF?)

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

                    # update hours seen
                    n_hours += batch_hours

                    pbar.update(batch_hours)

                    if n_updates % args.validation_steps == 0:
                        stop_training = False

                        if old_centers is not None:
                            delta = np.linalg.norm(kmeans.cluster_centers_ - old_centers) / np.linalg.norm(old_centers)
                            print(f"Centroid delta at step {n_updates}: {delta:.6f}")
                            if delta < args.centroid_delta:
                                print(f"Stopping: centroid delta below threshold ({delta} < {args.centroid_delta})")
                                stop_training = True
                        old_centers = kmeans.cluster_centers_.copy()

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
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--init", default="k-means++", choices=["k-means++", "random"])
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=0, type=int)
    parser.add_argument("--reassignment_ratio", default=0.01, type=float)  # resetting default to avoid dead centers
    parser.add_argument("--verbose", type=int, default=1)
    # everything below: minibatch-specific arguments
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--centroid-delta", type=float, default=1e-4)
    parser.add_argument("--validation-dir", nargs="*", help="Location of shards for validation (should be kept disjoint from training)")
    parser.add_argument("--inertia-tol", type=float, default=1e-3)
    parser.add_argument("--inertia-patience", type=int, default=5)
    args = parser.parse_args()
    main(args)
