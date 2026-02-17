"""
Inputs: 1) precomputed sonar embeddings of audio (from save-sonar-features.py)
        2) text that is aligned to these embeddings

Compute sonar embeddings for the text and then QE scores on top.
"""

import argparse
from os.path import join
from functools import partial
import json

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import soundfile as sf

from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


def load_translations(path):
    with open(path) as f:
        return [line.strip() for line in f]


def main(args):
    device = torch.device(args.device)

    text_embedder = TextToEmbeddingModelPipeline(
        encoder=args.text_encoder, tokenizer=args.text_tokenizer, device=device
    )
    blaser_qe = load_blaser_model(args.blaser_model).to(device).eval()

    # now, compute stuff
    translation_dataset = load_dataset(
        args.dataset_path, args.dataset_path_extra
    )[args.split].skip(args.start_ix)
    if args.n_examples > 0:
        examples_to_take = min(len(translation_dataset), args.n_examples)
        translation_dataset = translation_dataset.take(examples_to_take)
    translations = translation_dataset[args.translation_column]

    audio_sonar = np.load(args.audio_sonar)

    with torch.no_grad():
        scores = []

        for i in tqdm(range(0, len(translations), args.batch_size)):
            text = translations[i: i + args.batch_size]

            text_vecs = text_embedder.predict(
                text, batch_size=args.batch_size, source_lang=args.lang
            )
            speech_vecs = torch.from_numpy(audio_sonar[i: i + args.batch_size]).to(device)

            batch_scores = torch.nan_to_num(
                blaser_qe(src=speech_vecs, mt=text_vecs)
            ).squeeze().tolist()
            scores.extend(batch_scores)

    np.save(args.out_path, np.array(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blaser-model", default="blaser_2_0_qe")
    parser.add_argument("--text-encoder", default="text_sonar_basic_encoder")
    parser.add_argument("--text-tokenizer", default="text_sonar_basic_encoder")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-path-extra", nargs='?', const='')
    parser.add_argument("--split", default="train")
    parser.add_argument("--translation-column", default="mt")
    parser.add_argument("--audio-sonar",
                        help="path to .npy file containing sonar embeddings of audio examples")
    parser.add_argument("--start-ix", type=int, default=0,
                        help="For slicing an HF dataset (start index in the corpus)")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="For slicing an HF dataset (number of examples to take, starting with start-ix)")
    parser.add_argument("--lang", default="eng_Latn")
    parser.add_argument("--out_path")
    args = parser.parse_args()
    main(args)
