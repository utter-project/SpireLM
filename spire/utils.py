from os.path import exists, join, basename, dirname


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
