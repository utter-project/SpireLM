import argparse
from functools import partial
from os.path import dirname, basename, join, exists
from datasets import load_dataset

"""
Input should be sufficient info to load the HF dataset, output is a TSV file with
paths to the files in native format (i.e. wav for fleurs, flac for librispeech, etc.).

The idea is then either that this tsv is sufficient to produce the hubert features
(if it's wav), or that it gives you filenames that can be transformed to wav (otherwise)
"""


def fix_librispeech_path(ex_dict, path_extra, split):
    """
    This seems to work
    """
    alleged_path = ex_dict["file"]
    if exists(alleged_path):
        return alleged_path

    dir_part = dirname(alleged_path)

    path_split = "dev" if split == "validation" else split

    new_path = join(
        dir_part,
        "LibriSpeech", "-".join([path_split, path_extra]),
        str(ex_dict["speaker_id"]),
        str(ex_dict["chapter_id"]),
        str(ex_dict["id"]) + ".flac"
    )

    return new_path


def fix_fleurs_path(ex_dict, split):
    alleged_path = ex_dict["path"]
    if exists(alleged_path):
        return alleged_path

    path_split = "dev" if split == "validation" else split

    return join(dirname(alleged_path), path_split, basename(alleged_path))


def main(args):

    # which column contains the files?

    '''
    if args.dataset == "fleurs":
        key = "path"
        dataset = load_dataset("google/fleurs", "en_us", split="test", trust_remote_code=True)
    elif args.dataset == "ls-clean":
        key = "file"
        dataset = load_dataset("openslr/librispeech_asr", "clean", split="test", trust_remote_code=True)
    elif args.dataset == "ls-other":
        key = "file"
        dataset = load_dataset("openslr/librispeech_asr", "other", split="test", trust_remote_code=True)
    else:
        key = "audio"  # but then path within that
        dataset = load_dataset("facebook/voxpopuli", "en", split="test", trust_remote_code=True)
    '''

    path_funcs = {
        "facebook/voxpopuli": lambda ex_dict: ex_dict["audio"]["path"],
        "google/fleurs": partial(fix_fleurs_path, split=args.split),
        "openslr/librispeech_asr": partial(fix_librispeech_path, path_extra=args.path_extra, split=args.split)
    }

    path_func = path_funcs[args.path]

    dataset = load_dataset(args.path, args.path_extra, split=args.split, trust_remote_code=True)

    print("/")  # parent dir line of tsv
    for example_dict in dataset:
        # get the number of samples
        n_samples = example_dict["audio"]["array"].shape[0]
        path = path_func(example_dict)
        print("\t".join([path, str(n_samples)]))

    # iterate through dataset, returning the filenames and lengths

    # voxpopuli: d[0]["audio"]["path"]
    # [ex_dict["path"] for ex_dict in d["audio"]]
    # (verified that the paths are correct)
    # d[0]["audio"]["array"].shape[0] should give the second field of the tsv

    # librispeech:
    # d[0]["file"] gives an incorrect path that will look like
    # "/mnt/scratch-artemis/bpop/cache/datasets/downloads/extracted/f21f1c6ba595077b1ad806b535c4b0eda4b613263f221e2847227f11ea2b3223/6930-75918-0000.flac"
    # /mnt/scratch-artemis/bpop/cache/datasets/downloads/extracted/f21f1c6ba595077b1ad806b535c4b0eda4b613263f221e2847227f11ea2b3223/LibriSpeech/test-clean/
    # d[0]["audio"]["array"].shape[0] for second field


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", choices=["ls-clean", "ls-other", "voxpopuli", "fleurs"])
    parser.add_argument("--path", choices=["google/fleurs", "openslr/librispeech_asr", "facebook/voxpopuli"])
    parser.add_argument("--path-extra")  # generally a language or clean/other
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args)
