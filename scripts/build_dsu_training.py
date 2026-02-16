"""
Inputs:
    - text file containing DSUs
    - HF dataset (fine to remove audio if present); needs to also work for SPITE

Outputs:
    - jsonl file (support other formats? maybe later)
"""

import argparse
import json
from os.path import basename, splitext
from functools import partial
import random

from tqdm import tqdm
import numpy as np
from datasets import load_dataset, disable_caching
from spire.data import load_hf_audio_dataset
from spire.utils import load_template


def iterate_over_files(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                yield line


def load_librispeech_pc_transcripts(transcripts, transcript_column="text"):
    id2transcript = dict()
    with open(transcripts) as f:
        for line in f:
            transcript_metadata = json.loads(line)
            audio_id = splitext(basename(transcript_metadata['audio_filepath']))[0]
            # audio_id = transcript_metadata['id']
            id2transcript[audio_id] = transcript_metadata[transcript_column]
    return id2transcript


def load_simple_transcripts(transcripts, transcript_column="text"):
    out_transcripts = []
    with open(transcripts) as f:
        for line in f:
            transcript_metadata = json.loads(line)
            out_transcripts.append(transcript_metadata[transcript_column])
    return out_transcripts


def add_external_transcripts(dataset, transcript_path, transcript_in_column="text", librispeech_pc=False):

    # this block: librispeech-pc case
    if librispeech_pc:
        id2transcript = load_librispeech_pc_transcripts(transcript_path, transcript_column=transcript_in_column)
        transcripts = [id2transcript.get(ex_id, "") for ex_id in dataset["id"]]  # hardcoded id column
    else:
        # simpler, complete transcript loading
        transcripts = load_simple_transcripts(transcript_path, transcript_column=transcript_in_column)

    dataset = dataset.add_column("external_transcript", transcripts)
    dataset = dataset.filter(lambda ex: len(ex["external_transcript"]) > 0)
    return dataset


def make_instruction(dsu_seq, transcript, speech_turn="Speech: {dsu_seq}\nEnglish:"):
    out_dict = {
        "conversations": [{"from": "human", "value": speech_turn.format(dsu_seq=dsu_seq)},
                          {"from": "gpt", "value": transcript}]
    }
    return json.dumps(out_dict, ensure_ascii=False)


def keep_example(ex, min_columns=(), min_column_values=(), max_columns=(), max_column_values=()):
    assert len(min_columns) == len(min_column_values)
    assert len(max_columns) == len(max_column_values)

    return all(ex[col] >= val for col, val in zip(min_columns, min_column_values)) \
        and all(ex[col] <= val for col, val in zip(max_columns, max_column_values))


def absolute_filter(dataset, min_columns=(), min_column_values=(), max_columns=(), max_column_values=()):
    assert len(min_columns) == len(min_column_values)
    assert len(max_columns) == len(max_column_values)
    if not min_columns and not max_columns:
        return dataset

    filter_func = partial(
        keep_example,
        min_columns=args.min_columns,
        min_column_values=args.min_column_values,
        max_columns=args.max_columns,
        max_column_values=args.max_column_values
    )
    dataset = dataset.filter(filter_func)
    return dataset


def topk_filter(dataset, k=0, column=None, mode="highest"):
    if k == 0 or column is None or len(dataset) <= k:
        return dataset

    if column != "random":
        return dataset.sort(column, reverse=mode == "highest").take(k)

    sampled_indices = random.sample(range(len(dataset)), k)
    return dataset.select(sampled_indices)


def compute_stats(spite_dataset):
    """
    The purpose of this is to keep track of how much data remains after the
    filtering stage
    """
    qe = np.array(spite_dataset["cometqe_22"])
    qe_mean = qe.mean()

    xcomet_xl = np.array(spite_dataset["xcomet_xl"])
    xcomet_xl_mean = xcomet_xl.mean()

    blaser2_src = np.array(spite_dataset["blaser2_src"])
    blaser2_src_mean = blaser2_src.mean()

    blaser2_mt = np.array(spite_dataset["blaser2_mt"])
    blaser2_mt_mean = blaser2_mt.mean()

    length = np.array(spite_dataset["audio_length"])
    length_mean = length.mean()
    length_std = length.std()
    length_sum_hours = length.sum() / 3600
    return {"cometqe_22": qe_mean,
            "xcomet_xl": xcomet_xl_mean,
            "blaser2_src": blaser2_src_mean,
            "blaser2_mt": blaser2_mt_mean,
            "length_total_h": length_sum_hours,
            "length_mean_s": length_mean,
            "length_std_s": length_std}


def main(args):
    random.seed(a=args.seed)

    speech_turn = load_template(args.templates, args.template_key)

    if args.audio_dataset:
        disable_caching()
        dataset = load_hf_audio_dataset(
            args.dataset_path,
            path_extra=args.path_extra,
            split=args.split,
            from_disk=True,
            remove_audio=True
        )
    else:
        # for spite
        dataset = load_dataset(
            args.dataset_path, args.path_extra, trust_remote_code=True
        )[args.split]

    # iterate_over_files allows this to handle multiple shards. It's easier
    # if it's all already combined but I guess this is a nice generalization.
    dsus = [dsu_seq.strip() for dsu_seq in iterate_over_files(args.dsus)]
    dataset = dataset.add_column("dsus", dsus)
    print(dataset)

    # now: add external transcripts (if relevant)
    if args.external_transcripts is not None:
        # adding external transcripts may also mean filtering
        dataset = add_external_transcripts(
            dataset,
            args.external_transcripts,
            transcript_in_column=args.external_transcript_column,
            librispeech_pc="librispeech-pc" in args.external_transcripts)

    # filter first with absolute minimum/maximum examples
    dataset = absolute_filter(
        dataset,
        min_columns=args.min_columns,
        min_column_values=args.min_column_values,
        max_columns=args.max_columns,
        max_column_values=args.max_column_values
    )

    # filter second by taking the top values according to some column
    dataset = topk_filter(
        dataset,
        k=args.topk_examples,
        column=args.topk_column,
        mode=args.topk_mode
    )

    # statistics about whatever portion of the data is left after filtering
    if args.spite_stats is not None:
        stats = compute_stats(dataset)
        with open(args.spite_stats, "w") as f:
            json.dump(stats, f)

    # now: iterate over examples, turn them into prompts, and write them to
    # the output file
    with open(args.instructions, "w") as out_f:
        for ex in tqdm(dataset):
            instruction = make_instruction(
                ex["dsus"],
                ex[args.transcript_column],
                speech_turn=speech_turn
            )
            out_f.write(instruction + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsus", nargs="+")
    parser.add_argument("--templates", default=None, help="json prompt template")
    parser.add_argument("--template-key", default="template")
    parser.add_argument("--audio-dataset", action="store_true")
    parser.add_argument("--dataset-path")
    parser.add_argument("--path-extra")
    parser.add_argument("--split")
    parser.add_argument("--transcript-column", default="text")  # "raw_text" for VoxPopuli
    parser.add_argument("--external-transcript-column", default="text")
    parser.add_argument("--external-transcripts", help="For LibriSpeech-PC and People's Speech")
    parser.add_argument("--instructions", default="instructions.jsonl")
    parser.add_argument("--min-columns", nargs="*", default=[])
    parser.add_argument("--min-column-values", nargs="*", default=[], type=float)
    parser.add_argument("--max-columns", nargs="*", default=[])
    parser.add_argument("--max-column-values", nargs="*", default=[], type=float)
    parser.add_argument("--topk-examples", default=0, type=int)
    parser.add_argument("--topk-column", default=None)
    parser.add_argument("--topk-mode", default="highest", choices=["highest", "lowest"])
    parser.add_argument("--spite-stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
