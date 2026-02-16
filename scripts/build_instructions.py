"""
Convert raw inputs (meaning text or DSU sequences, not waveforms) into
instructions with tokenizer-specific formatting and save them to a jsonl file
where each line's only key is "instruction".
"""

import argparse
import json
from itertools import repeat
from contextlib import ExitStack

from transformers import AutoTokenizer

from spire.utils import load_template


def main(args):
    assert (len(args.src) + len(args.src_constants)) == len(args.src_names)
    template = load_template(args.templates, args.template_key)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    with ExitStack() as stack, open(args.out, "w") as outf:
        files = [stack.enter_context(open(f)) for f in args.src]
        files.extend([repeat(constant) for constant in args.src_constants])
        for src_cols in zip(*files):
            # use src_names to match content of files to slots in the template
            template_args = dict(zip(args.src_names, src_cols))
            prompt = template.format(**template_args)

            # convert prompt into format expected by LM tokenizer
            messages = [{"role": "user", "content": prompt}]
            tokenized_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outf.write(json.dumps({"instruction": tokenized_prompt}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", required=True,
                        help="""Paths to files to be used to construct prompts.
                        There needs to be at least one of these. You might have
                        more if, for example, the prompt includes few-shot
                        examples.""")
    parser.add_argument("--src-names", nargs="+", required=True,
                        help="""Names identifying which slot each src file fills.
                        For example, if the template has one slot called 'dsu_seq',
                        then --src-names dsu_seq should be passed. If src-constants
                        are present, they should be placed at the end of the
                        src names.""")
    parser.add_argument("--src-constants", nargs="*", default=[],
                        help="""Values that should be placed in every instruction
                        (for example, "--src-constants English" would make sense
                        if every example is a from-English translation). This is
                        equivalent to passing a file to --src that has the same
                        value in every line.""")
    parser.add_argument("--out", required=True, help="Path to save instructions to")
    parser.add_argument("--templates", default=None, help="json prompt template")
    parser.add_argument("--template-key", default="template")
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    main(args)
