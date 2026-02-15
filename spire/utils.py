from os.path import join
from glob import glob
import unicodedata
import json

import numpy as np


PRIVATE_OFFSET = 983040


def is_private_character(char):
    return unicodedata.category(char) == "Co"


def extra_id(i):
    return "<extra_id_{}>".format(str(i))


def pua(i):
    private_char = chr(int(i) + PRIVATE_OFFSET)
    assert is_private_character(private_char), i
    return private_char


def depua(char):
    assert is_private_character(char), char
    i = ord(char) - PRIVATE_OFFSET
    return i


def indices2dsus(indices, dsu_format="pua"):
    dsu_formatter = pua if dsu_format == "pua" else extra_id
    return "".join([dsu_formatter(i) for i in indices])


def deduplicate(tokens):
    out_tokens = []
    last_token = None
    for t in tokens:
        if t != last_token:
            out_tokens.append(t)
            last_token = t
    return out_tokens


def detokenize(labels, indices_only=False, deduplicated=True):
    labels = labels.to("cpu").tolist()  # should be a list of lists
    labels = [[l for l in lab if l != -1] for lab in labels]
    if deduplicated:
        labels = [deduplicate(lab) for lab in labels]

    if indices_only:
        return labels
    labels = [indices2dsus(ix) for ix in labels]
    return labels


def load_features(feat_files=None, feat_dir=None, n_files=0):
    assert feat_dir is not None or feat_files is not None
    assert feat_dir is None or feat_files is None

    if feat_dir is not None:
        feat_files = glob(join(feat_dir, "*.npy"))
        if n_files > 0:
            feat_files = sorted(feat_files)[:n_files]

    return np.vstack([np.load(p) for p in feat_files])


def load_template(template_path, key):
    if template_path is not None:
        with open(template_path) as f:
            speech_turn = json.load(f)[key]
    else:
        # Minimalistic default prompt, which is ambiguous between English ASR
        # and to-English ST
        speech_turn = "Speech: {dsu_seq}\nEnglish:"
    return speech_turn
