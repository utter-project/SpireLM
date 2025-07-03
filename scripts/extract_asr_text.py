import argparse
import json
from datasets import load_dataset

from spire.data import load_hf_audio_dataset


def main(args):
    # we don't want this
    # dataset = load_dataset(args.path, args.path_extra, split=args.split, trust_remote_code=True)

    # add from_disk)
    dataset = load_hf_audio_dataset(
        args.path, path_extra=args.path_extra, split=args.split, from_disk=args.dataset_type=="hf-disk"
    ).remove_columns("audio")

    text = dataset[args.text_field]
    with open(args.corpus, "w") as corpus_f:
        for line in text:
            corpus_f.write(line.strip() + "\n")

    metadata = []
    for example_dict in dataset:
        # del example_dict["audio"]
        metadata.append(example_dict)
    with open(args.metadata, "w") as metadata_f:
        json.dump(metadata, metadata_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", choices=["ls-clean", "ls-other", "voxpopuli", "fleurs"])
    # parser.add_argument("--path", choices=["google/fleurs", "openslr/librispeech_asr", "facebook/voxpopuli"])
    parser.add_argument("--path", required=True)
    parser.add_argument("--path-extra", default="")  # generally a language or clean/other
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-type", default="tsv", choices=["tsv", "hf-disk", "hf-cache"])
    parser.add_argument("--text-field", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    main(args)
