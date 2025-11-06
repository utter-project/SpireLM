import argparse
from tqdm import tqdm

import torch

from spire.utils import detokenize
from spire.hubert_labeler import HubertLabeler
from spire.data import build_dataloader


def pred2str_single(pred):
    if isinstance(pred, list):
        labels = [str(label) for label in pred]
        pred = " ".join(labels)
    return pred


def main(args):
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    labeler = HubertLabeler(
        args.ckpt_path, args.km_path, layer=args.layer, dtype=dtype
    )

    device = "cpu" if args.cpu else "cuda"
    labeler = labeler.to(device=device)
    labeler.eval()
    if args.compile:
        labeler = torch.compile(labeler)

    loader, n_batches, raw_length = build_dataloader(
        path=args.tsv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        validate_examples=args.validate_examples,
        path_extra=args.path_extra,
        hf_split=args.hf_split,
        resample_to=args.resample_to,
        hf_location="disk" if args.dataset_type == "hf-disk" else "cache"
    )

    with open(args.out_path, "w") as f:
        with torch.no_grad():
            labels = []
            indices = []
            for batch in tqdm(loader, total=n_batches):
                inp = batch.input_values.to(dtype=dtype, device=device)
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

        idx2labels = dict(zip(indices, labels))
        for i in range(raw_length):
            if i not in idx2labels:
                # this should be extremely rare
                f.write("\n")
                continue
            label_string = pred2str_single(idx2labels[i])
            f.write(label_string + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path")  # replace this
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ckpt_path", default="facebook/hubert-large-ll60k")
    parser.add_argument("--km_path", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--as-indices", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resample-to", type=int, default=16000)
    parser.add_argument("--dataset-type", default="tsv", choices=["tsv", "hf-disk", "hf-cache"])
    parser.add_argument("--path-extra", default="",
                        help="'xl' for Gigaspeech, for example")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--start-ix", type=int, default=0,
                        help="For slicing an HF dataset (start index in the corpus)")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Number of examples to take, starting with start-ix")
    parser.add_argument("--validate-examples", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="only useful for debugging")
    args = parser.parse_args()

    main(args)
