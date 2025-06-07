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


def pred2str(pred):
    return [pred2str_single(p) for p in pred]


def main(args):
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    # should deduplicated be an instance attribute or a labeling parameter?
    """
    labeler = Labeler(
        args.ckpt_path, args.km_path, feature_layer=args.layer,
        deduplicated=not args.no_dedup, dtype=dtype
    )
    """
    labeler = HubertLabeler(
        args.ckpt_path, args.km_path, layer=args.layer, dtype=dtype
    )

    labeler = labeler.to(device="cuda")
    labeler.eval()
    if args.compile:
        labeler = torch.compile(labeler)

    loader, n_batches = build_dataloader(
        path=args.tsv_path,
        sample_rate=16000,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        start_ix=args.start_ix,
        n_examples=args.n_examples
    )

    with open(args.out_path, "w") as f:
        with torch.no_grad():
            labels = []
            indices = []
            for batch in tqdm(loader, total=n_batches):
                inp = batch.input_values.to(dtype=dtype, device="cuda")
                mask = batch.attention_mask.cuda()
                # batch_labels = labeler.label(batch=inp, indices_only=args.as_indices, attention_mask=mask)
                batch_labels = labeler.predict(batch=inp, attention_mask=mask)
                detokenized_labels = detokenize(
                    batch_labels,
                    indices_only=args.as_indices,
                    deduplicated=not args.no_dedup
                )

                # total_tokens = inp.numel()
                # nonpad = mask.sum().item()
                # print(inp.shape, nonpad / total_tokens)

                if args.dataset_type == "tsv":

                    labels.extend(detokenized_labels)
                    if hasattr(batch, "indices"):
                        indices.extend(batch.indices)
                else:
                    for p in pred2str(detokenized_labels):
                        f.write(p + "\n")

        if args.dataset_type == "tsv":
            predictions = pred2str([label for i, label in sorted(zip(indices, labels))])
            for pred in predictions:
                f.write(pred + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--ckpt_path", default="facebook/hubert-large-ll60k")
    parser.add_argument("--km_path", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--as-indices", action="store_true")
    parser.add_argument("--legacy-audio", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dataset-type", default="tsv", choices=["tsv", "commonvoice", "spgi", "gigaspeech", "vctk"])
    parser.add_argument("--start-ix", type=int, default=0, help="For slicing an HF dataset (start index in the corpus)")
    parser.add_argument("--n-examples", type=int, default=0, help="For slicing an HF dataset (number of examples to take, starting with start-ix)")
    args = parser.parse_args()

    main(args)
