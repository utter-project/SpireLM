"""
Flores is no longer available as plain text, so we get to take a detour through
the wonderful world of Hugging Face.
"""

import argparse
from os import makedirs
from os.path import join

from datasets import load_dataset


def main(args):
    makedirs(args.dev, exist_ok=True)
    makedirs(args.devtest, exist_ok=True)

    for lang in args.langs:
        dev_path = "dev/{}.parquet".format(lang)
        devtest_path = "devtest/{}.parquet".format(lang)
        flores = load_dataset(
            args.flores_path,
            data_files={"dev": [dev_path], "devtest": [devtest_path]}
        )

        with open(join(args.dev, "dev." + lang), "w") as f:
            for line in flores["dev"]["text"]:
                f.write(line + "\n")

        with open(join(args.devtest, "devtest." + lang), "w") as f:
            for line in flores["devtest"]["text"]:
                f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["cmn_Hans", "deu_Latn", "eng_Latn", "fra_Latn", "ita_Latn", "kor_Hang", "nld_Latn", "rus_Cyrl", "por_Latn", "spa_Latn"])
    parser.add_argument("--flores-path", default="openlanguagedata/flores_plus")
    parser.add_argument("--dev", default="dev/")
    parser.add_argument("--devtest", default="devtest/")
    args = parser.parse_args()
    main(args)
