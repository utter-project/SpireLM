from os.path import exists, join, basename, dirname
import unicodedata

import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor


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


def load_wav(path, device, expected_sample_rate=None):
    feature_extractor = Wav2Vec2FeatureExtractor()
    wav, sample_rate = sf.read(path, dtype="float32")
    if expected_sample_rate is not None:
        assert sample_rate == expected_sample_rate, "Expected sample rate {}, got {}".format(expected_sample_rate, sample_rate)

    batch = feature_extractor(wav, sampling_rate=expected_sample_rate, return_tensors="pt").input_values.to(device)
    return batch


def detokenize(labels, indices_only=False, deduplicated=True):
    labels = labels.to("cpu").tolist()  # should be a list of lists
    labels = [[l for l in lab if l != -1] for lab in labels]
    if deduplicated:
        labels = [deduplicate(lab) for lab in labels]

    if indices_only:
        return labels
    labels = [indices2dsus(ix) for ix in labels]
    return labels


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
