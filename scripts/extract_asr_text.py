import argparse
import json

from spire.cli import dataset_parser
from spire.data import load_audio_dataset


def main(args):
    assert len(args.config) == 1
    dataset, _ = load_audio_dataset(args.config[0], remove_audio=True)

    text = dataset[args.text_field]
    with open(args.corpus, "w") as corpus_f:
        for line in text:
            corpus_f.write(line.strip() + "\n")

    metadata = []
    for example_dict in dataset:
        metadata.append(example_dict)
    with open(args.metadata, "w") as metadata_f:
        json.dump(metadata, metadata_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[dataset_parser])
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    main(args)
