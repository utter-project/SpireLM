import argparse
from tqdm import tqdm
import sys

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from spire.utils import load_features, batched_features


def train_in_memory(kmeans, args):
    features = load_features(feat_dir=args.feat_dir, feat_files=args.feat_files, n_files=args.n_files)
    kmeans.fit(features)
    return kmeans


def train_minibatch(kmeans, args):

    # lazy feature loading
    feature_batches = batched_features(
        args.batch_size,
        feat_dir=args.feat_dir,
        feat_files=args.feat_files,
        seed=args.seed
    )

    frames_seen = 0
    old_centers = None

    pbar_max = args.max_frames / 180000  # frames per hour
    with tqdm(total=pbar_max) as pbar:
        for i, batch in enumerate(feature_batches, 1):
            kmeans.partial_fit(batch)
            frames_seen += batch.shape[0]

            pbar.update(batch.shape[0] / 180000)
            if i % args.validation_steps == 0:
                stop_training = False

                if old_centers is not None:
                    delta = np.linalg.norm(kmeans.cluster_centers_ - old_centers) / np.linalg.norm(old_centers)
                    print(f"Centroid delta at step {i}: {delta:.6f}")
                    if delta < args.centroid_delta:
                        print(f"Stopping: centroid delta below threshold ({delta} < {args.centroid_delta})")
                        stop_training = True
                old_centers = kmeans.cluster_centers_.copy()

                sys.stdout.flush()
                if stop_training:
                    break

            if frames_seen >= args.max_frames:
                print(f"Stopping: reached max frames {args.max_frames}")
                break

    return kmeans


def main(args):
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        verbose=args.verbose,
        compute_labels=False,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        init_size=None,
        n_init=args.n_init if args.n_init > 0 else "auto",
        reassignment_ratio=args.reassignment_ratio,
        random_state=args.seed
    )

    if args.minibatch:
        kmeans = train_minibatch(kmeans, args)
    else:
        kmeans = train_in_memory(kmeans, args)

    joblib.dump(kmeans, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", nargs="*")
    parser.add_argument("--feat-files", nargs="*")
    parser.add_argument("--n-files", type=int, default=0, help="only with --feat-dir")
    parser.add_argument("--out-path", default="kmeans.joblib")
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--init", default="k-means++", choices=["k-means++", "random"])
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=0, type=int)
    parser.add_argument("--reassignment_ratio", default=0.01, type=float)  # resetting default to avoid dead centers
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--minibatch", action="store_true")
    # everything below: minibatch-specific arguments
    parser.add_argument("--max-frames", type=int, default=200_000_000)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--centroid-delta", type=float, default=1e-4)
    parser.add_argument("--validation-dir", nargs="*", help="Location of shards for validation (should be kept disjoint from training)")
    parser.add_argument("--inertia-tol", type=float, default=1e-3)
    parser.add_argument("--inertia-patience", type=int, default=5)
    args = parser.parse_args()
    main(args)
