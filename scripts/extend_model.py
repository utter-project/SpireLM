"""
Inputs: path to an existing llama model and a new vocabulary. The new
vocabulary should extend the llama vocabulary (i.e. the first V_0 entries
should be the same as the original llama tokenizer, where V_0 is the size of
the llama tokenizer).
"""

import argparse

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from transformers import LlamaForCausalLM, LlamaTokenizer


def embedding_normal(emb):
    # see https://www.cs.columbia.edu/~johnhew//vocab-expansion.html
    mu = torch.mean(emb, dim=0)
    n = emb.size(0)
    sigma = ((emb - mu).T @ (emb - mu)) / n
    dist = MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
    return dist


def main(args):
    # the new tokenizers I have are just sentencepiece models. How do I turn
    # them into LlamaTokenizers?

    model = LlamaForCausalLM.from_pretrained(args.model_path)
    orig_vocab_size = model.config.vocab_size

    new_tokenizer = LlamaTokenizer(args.new_tokenizer)
    new_vocab_size = len(new_tokenizer)
    assert new_vocab_size > orig_vocab_size

    # this handles both input and output embeddings
    model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=args.pad_multiple)

    num_new_types = new_vocab_size - orig_vocab_size

    if args.init_strategy == "mean":
        # set new embeddings
        with torch.no_grad():
            input_emb = model.model.embed_tokens.weight
            input_normal = embedding_normal(input_emb[:orig_vocab_size])
            new_input_emb = input_normal.sample((num_new_types,))
            model.model.embed_tokens.weight[orig_vocab_size:] = new_input_emb

            output_emb = model.lm_head.weight
            output_normal = embedding_normal(output_emb[:orig_vocab_size])
            new_output_emb = output_normal.sample((num_new_types,))
            model.lm_head.weight[orig_vocab_size:] = new_output_emb

    # save the model
    model.save_pretrained(args.out_dir)
    new_tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--new-tokenizer", help="path to spm model")
    parser.add_argument("--init-strategy", default="default", choices=["default", "mean"])
    parser.add_argument("--pad-multiple", type=int, default=64)
    opt = parser.parse_args()
    main(opt)
