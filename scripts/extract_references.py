import json
import argparse


def main(args):
    path = args.path
    key = args.key
    with open(path) as f:
        corpus = json.load(f)
    for example in corpus:
        line = example[key].strip()
        if args.strip:
            line = line.replace("English speech: ", "")
            line = line.replace(" \n English text:", "")
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--key")
    parser.add_argument("--strip", action="store_true")
    args = parser.parse_args()
    main(args)
