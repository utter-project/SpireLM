import argparse
import json

import numpy as np


def read_comet(path):
    # keys are paths to files, values are lists of examples
    with open(path) as f:
        return json.load(f)


def system_score(sents):
    mean_score = np.array([sent["COMET"] for sent in sents]).mean().item()
    return mean_score


def main(args):
    all_scores = read_comet(args.path)
    by_level = {k: system_score(v) for k, v in all_scores.items()}

    with open(args.mean_results, "w") as f:
        json.dump(by_level, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mean-results", required=True)
    args = parser.parse_args()
    main(args)
