import argparse

from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
from tqdm import tqdm
import joblib

from spire.hubert_labeler import Featurizer
from spire.data import build_dataloader


def main(args):

    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtypes[args.dtype]

    device = "cpu"

    torch_random = torch.Generator(device=device)
    torch_random.manual_seed(args.torch_seed)

    featurizer = Featurizer(args.ssl_model, layer=args.layer, dtype=dtype)

    featurizer = featurizer.to(device=device)
    featurizer.eval()
    if args.compile:
        featurizer = torch.compile(featurizer)

    # for now, I don't know what to do about batching. It's a problem for later
    loader, n_batches, raw_length = build_dataloader(
        path=args.data_path,
        batch_size=args.ssl_batch_size,
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        validate_examples=False,  # or we can try I guess
        path_extra=args.path_extra,
        hf_split=args.hf_split,
        resample_to=args.resample_to,
        hf_location="disk" if args.dataset_type == "hf-disk" else "cache",
        shuffle=True,
        torch_random=torch_random
    )

    # I think I don't need to use AutoFeatureExtractor here. The spire data
    # code should apply the Wav2Vec2FeatureExtractor, which as far as I know is
    # appropriate (but we can revisit this if it turns out to be wrong).
    # processor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    # model = Wav2Vec2BertModel.from_pretrained(args.ssl_model).to(device)
    # model.eval()

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        verbose=1,
        compute_labels=False
    )

    kmeans_buffer = []
    frame_count = 0
    kmeans_batch = args.kmeans_batch_size if args.kmeans_batch_size is not None else 10 * args.n_clusters
    n_hours = 0.

    with torch.no_grad():
        with tqdm(total=args.max_hours) as pbar:
            for batch in loader:
                inp = batch.input_values.to(dtype=dtype, device=device)

                mask = batch.attention_mask
                if device == "cuda":
                    mask = mask.cuda()
                features = featurizer(batch=inp, attention_mask=mask, flatten=True)
                # flatten=True removes padding positions

                kmeans_buffer.append(features.cpu().numpy())

                batch_frames = features.shape[0]
                batch_hours = mask.sum().item() / (args.resample_to * 3600)
                print("This batch:", batch_frames, batch_hours, inp_batch_hours)

                frame_count += batch_frames

                # update hours seen
                n_hours += batch_hours

                # approximate batch size...

                if frame_count >= kmeans_batch or n_hours >= args.max_hours:
                    print("running partial fit")
                    # second disjunct is so that
                    kmeans.partial_fit(np.vstack(kmeans_buffer))
                    kmeans_buffer = []
                    frame_count = 0

                pbar.update(batch_hours)

                if n_hours >= args.max_hours:
                    break

    # save model
    joblib.dump(kmeans, args.out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-model", default="facebook/hubert-large-ll60k",
                        help="also try others like facebook/w2v-bert-2.0")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--data-path", default="google/fleurs")
    parser.add_argument("--n-clusters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=43)
    parser.add_argument("--out-path", default="kmeans.joblib")
    parser.add_argument("--ssl-batch-size", type=int, default=1)
    parser.add_argument("--kmeans-batch-size", type=int, default=None)
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
    args = parser.parse_args()
    main(args)
