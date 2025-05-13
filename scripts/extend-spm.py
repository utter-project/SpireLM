"""
This script allows
"""

import argparse

from spire.tokenizer_extension import extend_spm, convert_spm_to_hf, add_instruct_extras
from spire.utils import indices2dsus


def main(args):

    new_specials = []
    if args.new_specials is not None:
        new_specials = args.new_specials.split(",")

    assert (args.n_new_dsus is not None) != (args.new_types is not None), \
        "Must provide one of --n_new_dsus and --new_types, but not both"
    new_types = []
    if args.new_types is not None:
        with open(args.new_types) as f:
            new_types.extend([line.strip() for line in f])
    else:
        new_types.extend(indices2dsus(range(args.n_new_dsus), args.dsu_format))

    # extend the sentencepiece model
    extend_spm(
        args.original,
        args.spm_prefix,
        new_specials,
        new_types,
        scoring_type=args.scoring,
        new_types_are_special=args.special
    )

    # make an HF tokenizer out of the just-extended spm tokenizer
    hf_model = convert_spm_to_hf(args.spm_prefix)
    hf_model.save_pretrained(args.hf_base)

    # this block: make the instruct tokenizer
    if args.hf_instruct is not None:
        hf_model = add_instruct_extras(hf_model)
        hf_model.save_pretrained(args.hf_instruct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', help="Path to original sentencepiece model; but do we want the HF tokenizer instead?")
    parser.add_argument("--n_new_dsus", type=int, help="Number of new DSUs to create (mutually exclusive with --new_types)")
    parser.add_argument('--new_types', help="Path to new types to create (mutually exclusive with --n_new_dsus)")
    parser.add_argument("--dsu_format", choices=["pua", "extra_id"], default="pua", help="If using --n_new_dsus, format they should take (default: 'pua' for private use area characters)")
    parser.add_argument('--new_specials', help="Comma-delimited list of new special types to add")
    parser.add_argument("--spm_prefix", required=True, help="Extended sentencepiece model will be saved to spm_prefix.{model, vocab}.")
    parser.add_argument("--hf_base", required=True, help="Path to save HF tokenizer for SpireBase")
    parser.add_argument("--hf_instruct", help="Path to save HF tokenizer for instruction-tuned Spire models (optional)")
    parser.add_argument("--scoring", default="bpe", choices=["bpe", "none"])
    parser.add_argument("--special", action="store_true",
                        help="Add if the types should be stored as specials (necessary if DSUs look like <extra_token_{i}>, unnecessary if using Private Use Area characters)")
    args = parser.parse_args()
    main(args)
