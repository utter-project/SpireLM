import argparse
import json

from evaluate import load


def load_corpus(path):
    with open(path) as f:
        return [line.strip() for line in f]


def main(args):
    wer = load('wer')

    hyp = load_corpus(args.hyp)
    ref = load_corpus(args.ref)

    results_wer = wer.compute(predictions=hyp, references=ref)
    results = {'wer': results_wer}
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp")
    parser.add_argument("--ref")
    args = parser.parse_args()
    main(args)
