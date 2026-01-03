import argparse
from glob import glob
from os.path import join

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from tqdm import tqdm
import joblib

from spire.data import build_dataloader


"""
Replace this with something else; we need to separate the featurization from
the kmeans learning (following fairseq/espnet)
"""


def main(args):

    # I think I don't need to use AutoFeatureExtractor here. The spire data
    # code should apply the Wav2Vec2FeatureExtractor, which as far as I know is
    # appropriate (but we can revisit this if it turns out to be wrong).
    # processor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    # model = Wav2Vec2BertModel.from_pretrained(args.ssl_model).to(device)
    # model.eval()

    """
    def get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    ):
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=1,
            compute_labels=False,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=None,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
        )
    """

    # should this be MiniBatch or the full one?
    '''
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        verbose=1,
        compute_labels=False
    )
    '''

    kmeans = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            init=args.init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            verbose=2,
            compute_labels=False,
            tol=args.tol,
            max_no_improvement=args.max_no_improvement,
            init_size=None,
            n_init=args.n_init,
            reassignment_ratio=args.reassignment_ratio,
        )

    # load the features
    # (glob the .npy files, take a fixed amount from them I guess)
    features = np.vstack([np.load(p) for p in glob(join(args.feat_dir, "*.npy"))])
    print(features.shape)
    features = features[:1000,:128]
    print(features.shape)

    kmeans.fit(features)

    # save model
    joblib.dump(kmeans, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir")  # load from this
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", default="kmeans.joblib")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    main(args)
