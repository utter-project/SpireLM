import argparse

from tqdm import tqdm
import numpy as np

from datasets import disable_caching

from spire.cli import dataset_parser
from spire.data import load_audio_dataset


def compute_time_from_array(example):
    example["seconds"] = example["audio"]["array"].shape[0] / example["audio"]["sampling_rate"]
    return example


def compute_spgi_time(example):
    example["seconds"] = example["wav_filesize"] / 32000  # I believe this is correct
    return example


def compute_gigaspeech_time(example):
    example["seconds"] = example["end_time"] - example["begin_time"]
    return example


def compute_mls_eng_time(example):
    example["seconds"] = example["audio_duration"]
    return example


def main(args):

    length_funcs = {
        "spgispeech": compute_spgi_time,
        "gigaspeech": compute_gigaspeech_time,
        "mls_eng": compute_mls_eng_time
    }
    remove_audio = args.compute_length_func in length_funcs
    compute_time_func = length_funcs.get(
        args.compute_length_func,
        compute_time_from_array
    )

    dataset, _ = load_audio_dataset(args.config, remove_audio=remove_audio)
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
    parser = argparse.ArgumentParser(parents=[dataset_parser])
    parser.add_argument("--compute-length-func", default=None)
    parser.add_argument("--out", default="times.npy")
    args = parser.parse_args()
    main(args)
