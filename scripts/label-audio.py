import argparse
from tqdm import tqdm

import torch

from spire.dsus import Labeler
from spire.data import build_dataloader


def main(args):
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    # should deduplicated be an instance attribute or a labeling parameter?
    labeler = Labeler(
        args.ckpt_path, args.km_path, feature_layer=args.layer,
        deduplicated=not args.no_dedup, dtype=dtype
    )

    loader, n_batches = build_dataloader(
        path=args.tsv_path,
        sample_rate=16000,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    with torch.no_grad():
        labels = []
        indices = []
        for batch in tqdm(loader, total=n_batches):
            inp = batch.input_values.to(dtype=dtype, device="cuda")
            mask = batch.attention_mask.cuda()
            batch_labels = labeler.label(batch=inp, indices_only=args.as_indices, attention_mask=mask)
            # total_tokens = inp.numel()
            # nonpad = mask.sum().item()
            # print(inp.shape, nonpad / total_tokens)
            labels.extend(batch_labels)
            indices.extend(batch.indices)

    predictions = [label for i, label in sorted(zip(indices, labels))]

    with open(args.out_path, "w") as f:
        for pred in predictions:
            if isinstance(pred, list):
                pred = " ".join([str(label) for label in pred])
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
    args = parser.parse_args()

    main(args)
