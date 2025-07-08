import argparse
from os.path import join


def main(args):
    # /mnt/scratch-artemis/bpop/SpireLM/experiments/label-commonvoice-outputs/Deduplicate/HubertDtype.fp32+ShardNumber.*/dsus.txt
    with open(args.out, "w") as outf:
        for i in range(args.shard_start, args.shard_end + 1):
            shard_path = join(args.corpus_dir, "HubertDtype.fp32+ShardNumber.{}".format(i), "dsus.txt")
            with open(shard_path) as f:
                for line in f:
                    outf.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="/mnt/scratch-artemis/bpop/SpireLM/experiments/label-commonvoice-outputs/Deduplicate")
    parser.add_argument("--out")
    parser.add_argument("--shard-start", type=int, default=1)
    parser.add_argument("--shard-end", type=int, default=83)
    args = parser.parse_args()
    main(args)
