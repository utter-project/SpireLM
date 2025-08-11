import json
import argparse
import random
from itertools import repeat

from tqdm import tqdm
import numpy as np


"""
Input looks like this:
{"src": "It is claimed that government officials had little time to react.",
 "mt": {"tower_instruct_mistral": {"mt": "Es wird...", "COMET": 0.88427734375},
        "euro_9B_instruct": {"mt": "Es wird...", "COMET": 0.884765625}}
}

Output looks like this:
{"conversations":
    [{"from": "human", "value": "Speech: 󰿯\nSpanish:"},
     {"from": "gpt", "value": "Y él desconocía..."}]}
"""


def select_mt(ex_dict, models=None):
    allowed_models = models if models is not None else ex_dict["mt"]
    chosen_model = max(allowed_models, key=lambda m: ex_dict["mt"][m]["COMET"])
    return chosen_model


def make_instruction(dsu_seq, translation, tgt_lang, speech_turn="Speech: {example}\n{tgt_lang}:"):
    prompt = speech_turn.format(example=dsu_seq, tgt_lang=tgt_lang)
    out_dict = {
        "conversations": [{"from": "human", "value": prompt},
                          {"from": "gpt", "value": translation}]
    }
    return json.dumps(out_dict, ensure_ascii=False)


def make_chosen_ex_dict(ex_dict, chosen_model):
    out_ex_dict = dict()
    for k, v in ex_dict.items():
        if k != "mt":
            out_ex_dict[k] = v
    for k, v in ex_dict["mt"][chosen_model].items():
        out_ex_dict[k] = v

    best_comet = ex_dict["mt"][chosen_model]["COMET"]

    best_models = []
    for k, v in ex_dict["mt"].items():
        # k is a model name, v is the dict containing mt and COMET
        if v["COMET"] == best_comet:
            best_models.append(k)

    out_ex_dict["chosen_model"] = best_models
    return out_ex_dict


# todo: options for multiple thresholds
def filter_by_threshold(mt_corpus, speech_corpus, threshold, models=None, audio_lengths=None):

    audio_lengths = np.load(audio_lengths) if audio_lengths is not None else repeat(None)

    with open(mt_corpus) as mt_inp_f, open(speech_corpus) as sp_inp_f:
        for mt_line, dsus, audio_length in tqdm(zip(mt_inp_f, sp_inp_f, audio_lengths)):
            dsus = dsus.strip()

            if not dsus:
                # ignore empty lines (some corpora contain them)
                continue

            ex_dict = json.loads(mt_line)
            selected_model = select_mt(ex_dict, models=models)
            selected_ex_dict = make_chosen_ex_dict(ex_dict, selected_model)
            if audio_length is not None:
                selected_ex_dict["audio_length"] = audio_length

            # translation and score
            mt = selected_ex_dict["mt"]
            comet = selected_ex_dict["COMET"]

            # absolute threshold
            if comet < threshold:
                continue

            yield {"dsu": dsus, "mt": mt, "ex_dict": selected_ex_dict}


def main(args):
    random.seed(a=args.seed)
    human_template = "Speech: {example}\n{tgt_lang}:"

    if args.models is not None:
        models = args.models.split(",")
    else:
        models = None

    examples = filter_by_threshold(
        args.mt_corpus,
        args.speech_corpus,
        args.threshold,
        models=models,
        audio_lengths=args.audio_lengths
    )

    # n_examples == 0 -> absolute threshold-based filtering
    if args.n_examples > 0:
        if args.subsampling == "topk":
            examples = sorted(
                examples, key=lambda ex: ex["ex_dict"]["COMET"],
                reverse=True
            )[:args.n_examples]
        else:
            examples = list(examples)
            random.shuffle(examples)
            examples = examples[:args.n_examples]

    with open(args.filtered_mt_corpus, "w") as out_f, open(args.metadata, "w") as metadata_f:
        for ex in examples:
            instruction = make_instruction(
                ex["dsu"], ex["mt"], args.tgt, speech_turn=human_template
            )
            out_f.write(instruction + "\n")
            metadata = ex["ex_dict"]
            metadata_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mt-corpus")
    parser.add_argument("--filtered-mt-corpus")
    parser.add_argument("--metadata")
    parser.add_argument("--speech-corpus")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--n-examples", type=int, default=0)
    parser.add_argument("--subsampling", choices=["random", "topk"])
    parser.add_argument("--models", default=None)
    parser.add_argument("--audio-lengths")
    parser.add_argument("--tgt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
