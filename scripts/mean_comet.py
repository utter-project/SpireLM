import argparse
import json

import numpy as np


def read_comet(path):
    # keys are paths to files, values are lists of examples
    with open(path) as f:
        return json.load(f)


def read_comet_jsonl(path):
    """
    Packing this into a dictionary is a kludge
    """
    with open(path) as f:
        return {"scores": [json.loads(line) for line in f]}


def system_score(sents):
    mean_score = np.array([sent["COMET"] for sent in sents]).mean().item()
    return mean_score


def main(args):
    if args.jsonl:
        all_scores = read_comet_jsonl(args.path)
    else:
        all_scores = read_comet(args.path)
    by_level = {k: system_score(v) for k, v in all_scores.items()}

    with open(args.mean_results, "w") as f:
        json.dump(by_level, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mean-results", required=True)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()
    main(args)
