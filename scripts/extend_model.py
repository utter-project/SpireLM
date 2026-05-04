"""
Inputs: path to an existing llama model and a new vocabulary. The new
vocabulary should extend the llama vocabulary (i.e. the first V_0 entries
should be the same as the original llama tokenizer, where V_0 is the size of
the llama tokenizer).

Trying now to abstract this out to non-llama models. It should work perfectly
fine, honestly, and the part where the problems might leak in is in the
extend_tokenizer script (since it currently assumes a sentencepiece-backed
tokenizer, even if it is subsequently converted to a fast tokenizer)
"""

import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib

from spire.cli import dsu_parser, randomness_parser


def embedding_normal(emb):
    # see https://www.cs.columbia.edu/~johnhew//vocab-expansion.html
    mu = torch.mean(emb, dim=0)
    n = emb.size(0)
    sigma = ((emb - mu).T @ (emb - mu)) / n
    dist = MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
    return dist


def mean_init(emb_matrix, orig_size, n_new):
    """Initialize new embeddings by sampling from a Gaussian fitted to the original embeddings."""
    dist = embedding_normal(emb_matrix[:orig_size])
    emb_matrix[orig_size:] = dist.sample((n_new,))
    return emb_matrix


def random_orthogonal_init(emb_matrix, orig_size, n_new, centroids, eps=1e-5, match_original_stats=True):
    """Initialize new embeddings by applying a random orthogonal linear transformation to the centroids."""
    if centroids is None:
        raise ValueError("Centroids must be provided for random orthogonal initialization.")

    K, d_c = centroids.shape
    linear = nn.Linear(d_c, emb_matrix.shape[1], bias=False)

    # orthogonal init
    nn.init.orthogonal_(linear.weight)

    E = linear(centroids)  # (K, d_model)

    if match_original_stats:
        orig_emb = emb_matrix[:orig_size]
        target_mean = orig_emb.mean(dim=0, keepdim=True)
        target_std = orig_emb.std(dim=0, keepdim=True)
        E = (E - E.mean(dim=0, keepdim=True)) / (E.std(dim=0, keepdim=True) + eps)
        E = E * target_std + target_mean
    else:
        # normalize norms
        norms = E.norm(dim=1, keepdim=True) + eps
        E = E / norms
    print("this is K", K)
    print("this is n_new", n_new)
    print("E shape", E.shape)
    print("emb_matrix shape", emb_matrix.shape)
    print("the slice size is", emb_matrix[orig_size: orig_size + K].shape)

    supposed_emb_matrix = emb_matrix[orig_size: orig_size + K]
    print("supposed_emb_matrix shape", supposed_emb_matrix.shape)
    emb_matrix[orig_size: orig_size + K] = E

    return emb_matrix


def main(args):
    torch.manual_seed(args.seed)  # mismatch from elsewhere in repo, where args.torch_seed is used

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    original_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    orig_vocab_size = len(original_tokenizer)
    print("Original vocab size (excluding unused padding types)", orig_vocab_size)

    # original_vocab_size is the index where the new types begin
    # or is it? For qwen, the new specials seem to go first
    orig_vocab_size += 5  # kludge for now

    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer)
    new_vocab_size = len(new_tokenizer)
    print("New vocab size", new_vocab_size)
    assert new_vocab_size > orig_vocab_size

    num_new_types = new_vocab_size - orig_vocab_size

    # this handles both input and output embeddings
    print("weight shape before resize", model.model.embed_tokens.weight.shape)
    model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=args.pad_multiple)
    print("weight shape after resize", model.model.embed_tokens.weight.shape)

    if args.kmeans_model is not None:
        centroids = torch.from_numpy(
            joblib.load(args.kmeans_model).cluster_centers_
        )
        print(centroids.shape)
        print(orig_vocab_size, new_vocab_size, num_new_types)
        assert centroids.shape[0] == num_new_types
    else:
        centroids = None
    rand_orth_init = partial(
        random_orthogonal_init,
        centroids=centroids,
        match_original_stats=args.match_original_stats
    )
    init_strategies = {
        "mean": mean_init,
        "random_orthogonal": rand_orth_init,
    }

    if args.init_strategy in init_strategies:
        init_fn = init_strategies[args.init_strategy]

        with torch.no_grad():
            model.model.embed_tokens.weight = init_fn(
                model.model.embed_tokens.weight, orig_vocab_size, num_new_types
            )
            model.lm_head.weight = init_fn(
                model.lm_head.weight, orig_vocab_size, num_new_types
            )
    # else nothing to do because resize_token_embeddings already does
    # the default initialization

    # save the model
    model.save_pretrained(args.out_dir)
    new_tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[dsu_parser, randomness_parser])
    parser.add_argument("--model-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--new-tokenizer", help="path to spm model")
    parser.add_argument("--init-strategy", default="default",
                        choices=["default", "mean", "random_orthogonal"])
    parser.add_argument("--no-match-original-stats", dest="match_original_stats", action="store_false")
    parser.add_argument("--pad-multiple", type=int, default=64)
    args = parser.parse_args()
    main(args)
