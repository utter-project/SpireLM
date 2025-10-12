import argparse
import numpy as np
from scipy.spatial.distance import cdist
import joblib
import matplotlib.pyplot as plt


def isolation(clusters, sorted_ix, freqs):
    """
    sorted_ix is sorted by frequency.

    Therefore, the scores are also sorted by
    """
    # freqs are heights, analogous to the heights of mountains
    sorted_clusters = clusters[sorted_ix]

    dist = cdist(sorted_clusters, sorted_clusters, metric="euclidean")

    dist[np.triu_indices(freqs.shape[0])] = float("inf")

    scores = dist.min(axis=1)
    # out[i] = scores

    # these scores are in order of frequency. How do I get the indices in the
    # original vocabulary sorted by isolation?
    scores_ = scores[sorted_ix.argsort()]
    new_ranking = np.argsort(-scores_)  # right? right


    return new_ranking


def main(args):
    freqs = np.load(args.freqs)  # frequencies of DSU types, assembled elsewhere
    sorted_ix = np.argsort(-freqs)

    # maybe it would make sense to interpolate frequencies and isolation scores
    if args.metric == "isolation":
        assert args.kmeans is not None, "kmeans needs to be specified"
        m = joblib.load(args.kmeans)
        clusters = m.cluster_centers_
        sorted_ix = isolation(clusters, sorted_ix, freqs)

    kept = sorted_ix[:args.vocab_size]
    for k in kept:
        print(k.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqs")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--metric", choices=["frequency", "isolation"])
    parser.add_argument("--kmeans", help="optional kmeans model", default="/home/bpop/utter/3datsets_combined_kmeans_5000")
    args = parser.parse_args()
    main(args)
