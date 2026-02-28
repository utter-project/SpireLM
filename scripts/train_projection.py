"""
We're training regularized linear regression, but the data is too big to solve
analytically. So instead we do it with SGD.

General idea:
-- select a batch of audio
-- get both the features and the labels
-- use the labels to index the LLM embedding matrix

Train until convergence
"""

import argparse
import os
from os.path import join

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor
from npy_append_array import NpyAppendArray

from spire.labeler import Labeler
from spire.data import build_dataloader


def main(args):
    assert args.batch_size == 1

    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    device = "cpu" if args.cpu else "cuda"

    torch_random = torch.Generator(device="cpu")  # DataLoader always uses CPU
    torch_random.manual_seed(args.torch_seed)

    llm_embeddings = torch.load(args.llm_embeddings).to(device=device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)

    labeler = Labeler(
        args.ssl_model,
        args.km_path,
        layer=args.layer,
        dtype=dtype,
        pooling_width=args.pooling_width,
        pooling_type=args.pooling_type
    )

    labeler = labeler.to(device=device)
    labeler.eval()
    if args.compile:
        labeler = torch.compile(labeler)

    projector = nn.Linear(
        labeler.featurizer.feature_dim, llm_embeddings.shape[1],
        bias=True,
        device=device,
        dtype=dtype
    )
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )

    # already set up for batch == 1
    loader, n_batches, raw_length = build_dataloader(
        path=args.data_path,
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        validate_examples=False,  # necessary for some datasets
        path_extra=args.path_extra,
        hf_split=args.hf_split,
        resample_to=args.resample_to,
        hf_location="disk" if args.dataset_type == "hf-disk" else "cache",
        shuffle=True,
        torch_random=torch_random,
        pin_memory=not args.cpu,
        filter_mic=args.filter_mic
    )

    # future idea: specify size of shard in *either* hours or frames
    # (but not both)

    n_hours = 0.

    input_field_name = feature_extractor.model_input_names[0]

    loss_buffer = []
    min_loss = float("inf")
    buffers_since_improvement = 0

    with tqdm(total=args.max_hours) as pbar:
        for batch in loader:
            inp = batch[input_field_name].to(dtype=dtype, device=device)

            mask = batch.attention_mask
            if device == "cuda":
                mask = mask.cuda()

            with torch.no_grad():
                # we need both the features and the labels
                features, pad_percent = labeler.featurizer(
                    batch=inp,
                    attention_mask=mask,
                    flatten=True,
                    return_pad_percent=True
                )
                dist = labeler.kmeans(features)
                labels = dist.argmin(dim=-1)

                # now use the labels to index the LLM embeddings
                targets = llm_embeddings[labels]

            # now the model
            optim.zero_grad()
            targets_hat = projector(features)
            batch_loss = criterion(targets_hat, targets)
            batch_loss.backward()
            optim.step()

            loss_item = batch_loss.item()

            if len(loss_buffer) < 100:
                loss_buffer.append(loss_item)
            else:
                buffer_avg = sum(loss_buffer) / len(loss_buffer)
                loss_buffer = []
                print(buffer_avg)
                if buffer_avg < min_loss:
                    min_loss = buffer_avg
                    buffers_since_improvement = 0
                else:
                    buffers_since_improvement += 1
            batch_hours = sum(batch["seconds"]) / 3600
            n_hours += batch_hours
            pbar.update(batch_hours)

            if buffers_since_improvement >= 20:
                optim.param_groups[0]["lr"] /= 10
                # batches_since_improvement = 0

            if optim.param_groups[0]["lr"] <= 1e-6 or n_hours >= args.max_hours:
                print("Reached stop", optim.param_groups[0]["lr"])
                break

    torch.save(projector, args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-model", default="facebook/hubert-large-ll60k",
                        help="also try others like facebook/w2v-bert-2.0")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--km_path", default="/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000")
    parser.add_argument("--data-path", default="google/fleurs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=43)
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
    parser.add_argument("--max-hours", type=float, default=1000.,
                        help="""If specified, number of hours to train
                             clustering on (otherwise uses whole dataset)""")
    parser.add_argument("--validate-examples", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="only useful for debugging")
    parser.add_argument("--pooling-width", type=int, default=1, help="1 recovers no pooling")
    parser.add_argument("--pooling-type", choices=["mean", "max"], default="mean")
    parser.add_argument("--filter-mic", default=None)
    parser.add_argument("--llm-embeddings", required=True,
                        help="""DSU embeddings from a trained model""")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--save-model", default="projection.pt")
    args = parser.parse_args()
    main(args)
