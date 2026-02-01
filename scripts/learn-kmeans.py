import argparse
from glob import glob
from os.path import join

import joblib
from sklearn.cluster import MiniBatchKMeans
import numpy as np

from spire.utils import load_features


def main(args):
    # idea: try standard KMeans in addition to minibatch
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

    features = load_features(feat_dir=args.feat_dir, feat_files=args.feat_files, n_files=args.n_files)

    kmeans.fit(features)

    joblib.dump(kmeans, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir")
    parser.add_argument("--feat-files", nargs="*")
    parser.add_argument("--n-files", type=int, default=0, help="only with --feat-dir")
    parser.add_argument("--out-path", default="kmeans.joblib")
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--init", default="k-means++", choices=["k-means++", "random"])
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=0, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    main(args)
