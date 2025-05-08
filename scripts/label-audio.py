import argparse
from tqdm import tqdm

from spire.dsus import Labeler


def main(args):
    # should deduplicated be an instance attribute or a labeling parameter?
    labeler = Labeler(
        args.ckpt_path, args.km_path, feature_layer=args.layer,
        kmeans_device="cuda:0", deduplicated=not args.no_dedup
    )

    with open(args.out_path, "w") as f:
        predictions = labeler.label_corpus(
            args.tsv_path,
            indices_only=args.as_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        for pred in predictions:
            if isinstance(pred, list):
                pred = " ".join([str(label) for label in pred])
            f.write(pred + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ckpt_path", default="facebook/hubert-large-ll60k")
    parser.add_argument("--km_path", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--as-indices", action="store_true")
    parser.add_argument("--legacy-audio", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    main(args)
