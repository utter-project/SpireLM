import argparse
import os
from os.path import join

from tqdm import tqdm
import torch
from npy_append_array import NpyAppendArray

from spire.hubert_labeler import Featurizer
from spire.data import build_dataloader


def main(args):

    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    device = "cpu" if args.cpu else "cuda"

    torch_random = torch.Generator(device="cpu")  # DataLoader always uses CPU
    torch_random.manual_seed(args.torch_seed)

    featurizer = Featurizer(args.ssl_model, layer=args.layer, dtype=dtype)

    featurizer = featurizer.to(device=device)
    featurizer.eval()
    if args.compile:
        featurizer = torch.compile(featurizer)

    loader, n_batches, raw_length = build_dataloader(
        path=args.data_path,
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
        pin_memory=not args.cpu
    )

    # I think I don't need to use AutoFeatureExtractor here. The spire data
    # code should apply the Wav2Vec2FeatureExtractor, which as far as I know is
    # appropriate (but we can revisit this if it turns out to be wrong).

    # (however, I should change this later when I make the code extensible)

    # processor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    # model = Wav2Vec2BertModel.from_pretrained(args.ssl_model).to(device)
    # model.eval()

    shard_size = 3600 * 50  # an hour of frames

    shard_frames = 0
    n_hours = 0.
    shard_index = 0

    os.makedirs(args.feature_dir, exist_ok=True)
    feat_f = NpyAppendArray(join(args.feature_dir, f"shard_{shard_index}.npy"))

    with torch.no_grad():
        with tqdm(total=args.max_hours) as pbar:
            for batch in loader:
                inp = batch.input_values.to(dtype=dtype, device=device)

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
                # flatten=True removes padding positions
                batch_frames = features.shape[0]
                batch_hours = mask.sum().item() / (args.resample_to * 3600)

                feat_f.append(features.cpu().float().numpy())

                shard_frames += batch_frames

                # update hours seen
                n_hours += batch_hours

                if shard_frames >= shard_size or n_hours >= args.max_hours:
                    shard_frames = 0

                    shard_index += 1
                    feat_f = NpyAppendArray(join(args.feature_dir, f"shard_{shard_index}.npy"))

                pbar.update(batch_hours)

                if n_hours >= args.max_hours:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-model", default="facebook/hubert-large-ll60k",
                        help="also try others like facebook/w2v-bert-2.0")
    parser.add_argument("--layer", type=int, default=22)
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
    parser.add_argument("--feature-dir")
    args = parser.parse_args()
    main(args)
