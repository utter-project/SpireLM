import argparse
from tqdm import tqdm

import torch
from transformers import AutoFeatureExtractor

from spire.utils import detokenize
from spire.labeler import Labeler
from spire.data import build_dataloader
from spire.cli import ssl_parser, dataset_parser, dsu_parser


def pred2str_single(pred):
    if isinstance(pred, list):
        labels = [str(label) for label in pred]
        pred = " ".join(labels)
    return pred


def main(args):
    assert len(args.config) == 1
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    labeler = Labeler(
        args.ssl_model,
        args.kmeans_model,
        layer=args.layer,
        keep_final_layer_norm=args.keep_final_layer_norm,
        dtype=dtype,
        pooling_width=args.pooling_width,
        pooling_type=args.pooling_type
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)

    device = "cpu" if args.cpu else "cuda"
    labeler = labeler.to(device=device)
    labeler.eval()
    if args.compile:
        labeler = torch.compile(labeler)

    loader, n_batches = build_dataloader(
        config=args.config,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        resample_to=args.resample_to,
        token_batching=args.token_batching,
        example_lengths=args.example_lengths
    )

    input_field_name = feature_extractor.model_input_names[0]

    with open(args.out_path, "w") as f:
        with torch.no_grad():
            labels = []
            indices = []
            lengths = []
            for batch in tqdm(loader, total=n_batches):
                inp = batch[input_field_name].to(dtype=dtype, device=device)
                mask = batch.attention_mask
                if device == "cuda":
                    mask = mask.cuda()
                batch_labels = labeler.predict(batch=inp, attention_mask=mask)
                detokenized_labels = detokenize(
                    batch_labels,
                    indices_only=args.as_indices,
                    deduplicated=not args.no_dedup
                )

                labels.extend(detokenized_labels)
                indices.extend(batch.indices)
                lengths.extend(batch.seconds)

        idx2labels = {i: (label, length) for i, label, length in zip(indices, labels, lengths)}
        for i in range(len(loader.dataset)):
            if i not in idx2labels:
                # this should be extremely rare
                f.write("\n")
                continue
            label, length = idx2labels[i]
            label_string = pred2str_single(label) if length > 0 else ""
            f.write(label_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[ssl_parser, dataset_parser, dsu_parser])
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()

    main(args)
