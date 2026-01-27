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

from tqdm import tqdm
from datasets import load_dataset
from spire.data import load_hf_audio_dataset


def iterate_over_files(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                yield line


def load_transcripts(transcripts, transcript_column="text"):
    id2transcript = dict()
    with open(transcripts) as f:
        for line in f:
            transcript_metadata = json.loads(line)
            audio_id = splitext(basename(transcript_metadata['audio_filepath']))[0]
            # audio_id = transcript_metadata['id']
            id2transcript[audio_id] = transcript_metadata[transcript_column]
    return id2transcript


def add_external_transcripts(dataset, transcript_path, transcript_column="text"):
    id2transcript = load_transcripts(transcript_path, transcript_column=transcript_column)
    # hardcoded id column
    transcripts = [id2transcript.get(ex_id, "") for ex_id in dataset["id"]]
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


def main(args):
    if args.template is not None:
        with open(args.template) as f:
            speech_turn = json.load(f)["template"]
    else:
        # Minimalistic default prompt, which is ambiguous between English ASR
        # and to-English ST
        speech_turn = "Speech: {dsu_seq}\nEnglish:"

    if args.audio_dataset:
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
        dataset = add_external_transcripts(dataset, args.external_transcripts)

    # now: filtering (if relevant)
    # also to consider: filtering invalid rows (as in LibriSpeech-PC)
    filter_func = partial(
        keep_example,
        min_columns=args.min_columns,
        min_column_values=args.min_column_values,
        max_columns=args.max_columns,
        max_column_values=args.max_column_values
    )
    dataset = dataset.filter(filter_func)

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
    parser.add_argument("--template", default=None, help="json prompt template")
    parser.add_argument("--audio-dataset", action="store_true")
    parser.add_argument("--dataset-path")
    parser.add_argument("--path-extra")
    parser.add_argument("--split")
    parser.add_argument("--transcript-column", default="text")  # "raw_text" for VoxPopuli
    parser.add_argument("--external-transcripts", help="For LibriSpeech-PC")
    parser.add_argument("--instructions", default="instructions.jsonl")
    parser.add_argument("--min-columns", nargs="*", default=[])
    parser.add_argument("--min-column-values", nargs="*", default=[], type=float)
    parser.add_argument("--max-columns", nargs="*", default=[])
    parser.add_argument("--max-column-values", nargs="*", default=[], type=float)
    args = parser.parse_args()
    main(args)
