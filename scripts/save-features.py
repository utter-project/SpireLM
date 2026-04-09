import argparse
import os
from os.path import join

from tqdm import tqdm
import torch
from transformers import AutoFeatureExtractor
from npy_append_array import NpyAppendArray

from spire.labeler import Featurizer
from spire.data import build_dataloader
from spire.cli import ssl_parser, dataset_parser, randomness_parser


def main(args):
    assert len(args.config) == 1
    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    device = "cpu" if args.cpu else "cuda"

    torch_random = torch.Generator(device="cpu")  # DataLoader always uses CPU
    torch_random.manual_seed(args.torch_seed)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)

    featurizer = Featurizer(
        args.ssl_model,
        layer=args.layer,
        dtype=dtype,
        pooling_width=args.pooling_width,
        pooling_type=args.pooling_type
    )

    featurizer = featurizer.to(device=device)
    featurizer.eval()
    if args.compile:
        featurizer = torch.compile(featurizer)

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

    # past: specified hour-long shards by multiplying hubert fps by 3600
    # shard_size = 3600 * 50  # an hour of frames
    # shard_frames = 0

    # present: directly specify number of *hours* per shard
    shard_hours = 0.

    # future idea: specify size of shard in *either* hours or frames
    # (but not both)

    n_hours = 0.
    shard_index = 0

    os.makedirs(args.feature_dir, exist_ok=True)
    feat_f = NpyAppendArray(join(args.feature_dir, f"shard_{shard_index}.npy"))

    input_field_name = feature_extractor.model_input_names[0]

    with torch.no_grad():
        with tqdm(total=args.max_hours) as pbar:
            for batch in loader:
                inp = batch[input_field_name].to(dtype=dtype, device=device)

                mask = batch.attention_mask
                if device == "cuda":
                    mask = mask.cuda()

                # reminder: featurizer is essentially the forward of a HubertModel,
                # but with some layers cut off (is there a more elegant way to
                # do this inside HF?)
                features, pad_percent = featurizer(
                    batch=inp,
                    attention_mask=mask,
                    flatten=True,
                    return_pad_percent=True
                )

                batch_hours = sum(batch["seconds"]) / 3600

                feat_f.append(features.cpu().float().numpy())

                # update hours seen
                shard_hours += batch_hours
                n_hours += batch_hours

                if shard_hours >= args.hours_per_shard or n_hours >= args.max_hours:
                    shard_hours = 0.

                    shard_index += 1
                    feat_f = NpyAppendArray(join(args.feature_dir, f"shard_{shard_index}.npy"))

                pbar.update(batch_hours)

                if n_hours >= args.max_hours:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[ssl_parser, dataset_parser, randomness_parser])
    parser.add_argument("--max-hours", type=float, default=1000.,
                        help="""If specified, number of hours to train
                             clustering on (otherwise uses whole dataset)""")
    parser.add_argument("--feature-dir")
    parser.add_argument("--hours-per-shard", type=float, default=1.)
    args = parser.parse_args()
    main(args)
