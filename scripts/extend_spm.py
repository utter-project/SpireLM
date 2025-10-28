"""
This script allows
"""

import argparse

from transformers import AutoTokenizer

from spire.tokenizer_extension import extend_tokenizer, add_instruct_extras
from spire.utils import indices2dsus


def read_new_types(path, original_tokenizer_path):
    original_vocab = AutoTokenizer.from_pretrained(original_tokenizer_path).vocab

    new_types = []
    with open(path) as f:
        for line in f:
            new_type = line.strip().split("\t")[0]
            if new_type not in original_vocab:
                new_types.append(new_type)
    return new_types


def main(args):

    new_specials = []
    if args.new_specials is not None:
        new_specials = args.new_specials.split(",")

    assert (args.n_new_dsus is not None) != (args.new_types is not None), \
        "Must provide one of --n_new_dsus and --new_types, but not both"
    new_types = []
    if args.new_types is not None:
        # problem: new_types may overlap with existing types
        new_types.extend(read_new_types(args.new_types, args.original))
    else:
        new_types.extend(indices2dsus(range(args.n_new_dsus), args.dsu_format))

    hf_model = extend_tokenizer(
        args.original,
        args.spm_prefix,
        new_specials,
        new_types,
        scoring_type=args.scoring,
        new_types_are_special=args.special
    )

    hf_model.save_pretrained(args.hf_base)

    # this block: make the instruct tokenizer
    if args.hf_instruct is not None:
        hf_model = add_instruct_extras(hf_model)
        hf_model.save_pretrained(args.hf_instruct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', help="Path to original model (HF or sentencepiece)")
    parser.add_argument("--n_new_dsus", type=int,
                        help="Number of new DSUs to create (mutually exclusive with --new_types)")
    parser.add_argument('--new_types',
                        help="Path to new types to create (mutually exclusive with --n_new_dsus)")
    parser.add_argument("--dsu_format", choices=["pua", "extra_id"], default="pua",
                        help="""If using --n_new_dsus, format they should take
                             (default: 'pua' for private use area characters)""")
    parser.add_argument('--new_specials', help="Comma-delimited list of new special types to add")
    parser.add_argument("--spm_prefix", required=True,
                        help="Extended spm model will be saved to spm_prefix.{model, vocab}.")
    parser.add_argument("--hf_base", required=True, help="Path to save HF tokenizer for SpireBase")
    parser.add_argument("--hf_instruct",
                        help="Path to save HF tokenizer for IT'd Spire models (optional)")
    parser.add_argument("--scoring", default="bpe", choices=["bpe", "none"])
    parser.add_argument("--special", action="store_true",
                        help="""Add if the types should be stored as specials
                                (necessary if DSUs look like <extra_token_{i}>,
                                unnecessary if using Private Use Area characters)""")
    args = parser.parse_args()
    main(args)
