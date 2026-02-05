import argparse
import random


# We need to refactor this so that it's possible to do topk as well
def main(args):
    random.seed(a=args.seed)
    with open(args.in_file) as f:
        instructions = f.readlines()
        random.shuffle(instructions)

    subsampled_size = min(args.n_lines, len(instructions))
    with open(args.out_file, "w") as f:
        for i in range(subsampled_size):
            f.write(instructions[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file")
    parser.add_argument("--out-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-lines", type=int, default=10000)
    args = parser.parse_args()
    main(args)
