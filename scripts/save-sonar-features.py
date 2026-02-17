"""
Compute sonar embeddings for an audio corpus. These can be used for various
multimodal sentence embedding-related tasks. The main case targeted in this
repository is for computing Blaser 2.0, a QE metric.

If computing Blaser 2.0, sonar embeddings of audio are computed in this file and
the remainder of the work (computing sonar embeddings of text, running the
regression head) are handled in compute_blaser2.py. The reason for separating
the audio computation is that it is FAR more costly than everything else -- by
saving the audio embeddings, you avoid having to recompute them later.
"""

import argparse
from functools import partial

from tqdm import tqdm
import numpy as np
import torch
from npy_append_array import NpyAppendArray

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

from spire.data import build_dataloader


def collate_audio(inputs, sampling_rate=16000):
    """
    Unlike the default collators in spire.data, we appear not to need to apply
    an HF feature extractor to the audio here. It suffices merely to cast the
    audio features to torch.
    """
    batch = dict()
    audios = [torch.from_numpy(inp["audio"]["array"]).unsqueeze(0)
              for inp in inputs]

    seconds = []
    audios_with_placeholders = []

    for audio in audios:
        # placeholders for missing audio will have length 0
        length = audio.shape[1] / sampling_rate
        seconds.append(length)

        if length > 0:
            audios_with_placeholders.append(audio)
        else:
            placeholder = torch.zeros_like(next(a for a in audios if a.shape[1] > 0))
            audios_with_placeholders.append(placeholder)

    batch["seconds"] = seconds

    batch["audio"] = audios_with_placeholders
    batch["indices"] = [inp["idx"] for inp in inputs]
    return batch


def main(args):
    device = "cpu" if args.cpu else "cuda"
    device = torch.device(device)
    speech_embedder = SpeechToEmbeddingModelPipeline(
        encoder=args.sonar_speech_encoder, device=device
    )

    collator = partial(collate_audio, sampling_rate=args.resample_to)
    loader, n_batches, raw_length = build_dataloader(
        path=args.data_path,
        feature_extractor=None,
        batch_size=args.batch_size,  # this should be a number of seconds
        num_workers=args.num_workers,
        dataset_type=args.dataset_type,
        start_ix=args.start_ix,
        n_examples=args.n_examples,
        validate_examples=False,
        path_extra=args.path_extra,
        hf_split=args.hf_split,
        resample_to=args.resample_to,
        hf_location="disk" if args.dataset_type == "hf-disk" else "cache",
        pin_memory=not args.cpu,
        collator=collator,
        token_batching=args.token_batching,
        example_lengths=args.example_lengths,
        placeholder_len=0
    )

    with torch.no_grad():
        vectors = []
        indices = []
        lengths = []
        for batch in tqdm(loader, total=n_batches):
            speech = [b.to(device) for b in batch["audio"] if b.shape[0] > 0]

            speech_vecs = speech_embedder.predict(speech, batch_size=len(speech)).cpu().numpy()

            vectors.extend([sv for sv in speech_vecs])
            indices.extend(batch["indices"])
            lengths.extend(batch["seconds"])

    vec_size = vectors[0].shape[0]
    missing_vector = np.full((1, vec_size), np.nan)

    out = NpyAppendArray(args.save_vectors)

    idx2vec = {i: (vec, length) for i, vec, length in zip(indices, vectors, lengths)}
    for i in range(raw_length):
        if i not in idx2vec:
            out.append(missing_vector)
            continue
        vec, length = idx2vec[i]
        if length > 0:
            vec = vec[np.newaxis, :]
        else:
            vec = missing_vector
        out.append(vec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonar-speech-encoder", default="sonar_speech_encoder_eng")
    parser.add_argument("--data-path", default="google/fleurs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resample-to", type=int, default=16000)
    parser.add_argument("--dataset-type", default="tsv", choices=["tsv", "hf-disk", "hf-cache"])
    parser.add_argument("--path-extra", default="",
                        help="'xl' for Gigaspeech, for example")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--start-ix", type=int, default=0,
                        help="For slicing an HF dataset (start index in the corpus)")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Number of examples to take, starting with start-ix")
    parser.add_argument("--cpu", action="store_true", help="only useful for debugging")
    parser.add_argument("--token-batching", action="store_true")
    parser.add_argument("--example-lengths", default=None)
    parser.add_argument("--save-vectors", default="sonar.npy")
    args = parser.parse_args()
    main(args)
