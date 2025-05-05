import argparse
from tqdm import tqdm

from spire.dsus import Labeler


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start: end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)
    return iterate, len(lines)


def main(args):
    labeler = Labeler(
        args.ckpt_path, args.km_path, feature_layer=args.layer,
        kmeans_device="cuda:0", legacy_audio=args.legacy_audio,
        deduplicated=not args.no_dedup
    )
    # generator, num = get_path_iterator(args.tsv_path, args.nshard, args.rank)

    # iterator = generator()
    with open(args.out_path, "w") as f:
        predictions = labeler.label_corpus(args.tsv_path, indices_only=args.as_indices, batch_size=args.batch_size, num_workers=args.num_workers)

        '''
        for path, nsample in tqdm(iterator, total=num):
            labels = labeler.label(path, indices_only=args.as_indices)
            if isinstance(labels, list):
                labels = " ".join([str(label) for label in labels])
            f.write(labels + "\n")
        '''
        for pred in predictions:
            if isinstance(pred, list):
                pred = " ".join([str(label) for label in pred])
            f.write(pred + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ckpt_path", default="/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt")
    parser.add_argument("--km_path", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
    parser.add_argument("--layer", type=int, default=22)  # 22
    parser.add_argument("--nshard", type=int, default=1)  # 1
    parser.add_argument("--rank", type=int, default=0)  # 0
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--as-indices", action="store_true")
    parser.add_argument("--legacy-audio", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    main(args)
