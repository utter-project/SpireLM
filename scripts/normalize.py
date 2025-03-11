import argparse
import sys

# importing whisper_normalizer and calling
# whisper_normalizer.english.EnglishTextNormalizer() throws an error
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer


def main(args):
    if args.normalizer == "english":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    for line in args.infile:
        # the newline is necessary because the normalizer strips it from its input
        args.outfile.write(normalizer(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--outfile", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--normalizer", choices=["basic", "english"], default="english")
    args = parser.parse_args()

    main(args)
