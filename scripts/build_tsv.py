import argparse
from functools import partial
from datasets import load_dataset

from spire.utils import fix_fleurs_path, fix_librispeech_path

"""
Input should be sufficient info to load the HF dataset, output is a TSV file with
paths to the files in native format (i.e. wav for fleurs, flac for librispeech, etc.).

The idea is then either that this tsv is sufficient to produce the hubert features
(if it's wav), or that it gives you filenames that can be transformed to wav (otherwise)
"""


def main(args):
    path_funcs = {
        "facebook/voxpopuli": lambda ex_dict: ex_dict["audio"]["path"],
        "google/fleurs": partial(fix_fleurs_path, split=args.split),
        "openslr/librispeech_asr": partial(fix_librispeech_path, path_extra=args.path_extra, split=args.split)
    }

    path_func = path_funcs[args.path]

    dataset = load_dataset(args.path, args.path_extra, split=args.split, trust_remote_code=True)

    with open(args.out, "w") as f:
        f.write("/\n")  # parent dir line of tsv
        for example_dict in dataset:
            # get the number of samples
            n_samples = example_dict["audio"]["array"].shape[0]
            path = path_func(example_dict)
            f.write("\t".join([path, str(n_samples)]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=["google/fleurs", "openslr/librispeech_asr", "facebook/voxpopuli"])
    parser.add_argument("--path-extra", help="generally a language or clean/other")
    parser.add_argument("--out")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args)
