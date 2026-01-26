import argparse

from tqdm import tqdm
import numpy as np

from datasets import disable_caching

from spire.data import load_hf_audio_dataset


def compute_commonvoice_time(example):
    example["seconds"] = example["audio"]["array"].shape[0] / example["audio"]["sampling_rate"]
    return example


def compute_time_from_array(example):
    example["seconds"] = example["audio"]["array"].shape[0] / example["audio"]["sampling_rate"]
    return example


def compute_spgi_time(example):
    example["seconds"] = example["wav_filesize"] / 32000  # I believe this is correct
    return example


def compute_gigaspeech_time(example):
    example["seconds"] = example["end_time"] - example["begin_time"]
    return example


def main(args):

    remove_audio = True
    if "spgispeech" in args.path:
        compute_time_func = compute_spgi_time
    elif "gigaspeech" in args.path:
        compute_time_func = compute_gigaspeech_time
    else:
        compute_time_func = compute_time_from_array
        remove_audio = False

    '''
    if "common_voice" in args.path:
        corpus_name = "commonvoice"
    elif "spgispeech" in args.path:
        corpus_name = "spgi"
    else:
        corpus_name = "gigaspeech"

    compute_time_funcs = {
        "commonvoice": compute_commonvoice_time,
        "spgi": compute_spgi_time,
        "gigaspeech": compute_gigaspeech_time
    }
    compute_time_func = compute_time_funcs[corpus_name]
    '''

    dataset = load_hf_audio_dataset(
        args.path, path_extra=args.path_extra, split=args.split,
        remove_audio=remove_audio
    )
    if remove_audio:
        disable_caching()
        dataset = dataset.map(compute_time_func)
        times = np.array(dataset["seconds"])
    else:
        times_list = []
        for ex in tqdm(dataset):
            ex = compute_time_from_array(ex)
            times_list.append(ex["seconds"])
        times = np.array(times_list)
    np.save(args.out, times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--path-extra", nargs="?", const="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="times.npy")
    args = parser.parse_args()
    main(args)
