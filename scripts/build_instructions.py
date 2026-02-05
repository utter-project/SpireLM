import argparse
import json
import random
from functools import partial

from transformers import AutoTokenizer


def read_shot_paths(src_path, tgt_path, max_src_len=None, max_tgt_len=None):
    examples = []
    with open(src_path) as srcf, open(tgt_path) as tgtf:
        for src_line, tgt_line in zip(srcf, tgtf):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if max_src_len is not None and len(src_line) > max_src_len:
                continue
            if max_tgt_len is not None and len(tgt_line) > max_tgt_len:
                continue
            examples.append((src_line, tgt_line))
    return examples


class ShotSampler:

    def __init__(self, src_path, tgt_path, src_lang, tgt_lang):
        self.shots = read_shot_paths(src_path, tgt_path)
        self._template = "{src_lang}: {example_src}\n{tgt_lang}: {example_tgt}"

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

    @property
    def template(self):
        return self._template

    def sample(self):
        return random.choice(self.shots)

    def sample_template(self):
        example_src, example_tgt = self.sample()
        return self.template.format(
            example_src=example_src,
            example_tgt=example_tgt,
            src_lang=self._src_lang,
            tgt_lang=self._tgt_lang
        )

    def sample_context(self, n=1):
        sampled = [self.sample_template() for i in range(n)]
        return "\n".join(sampled) + "\n"


class ASRShotSampler(ShotSampler):

    def __init__(self, src_path, tgt_path, max_src_len=None, max_tgt_len=None, model="tower"):
        self.shots = read_shot_paths(src_path, tgt_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
        if model == "tower":
            self._template = "Speech: {example_src}\n English: {example_tgt}"
        else:
            self._template = "[Speech] {example_src} \n[Text] <START Transcript> {example_tgt} <END>"

    def sample_template(self):
        example_src, example_tgt = self.sample()
        return self.template.format(
            example_src=example_src,
            example_tgt=example_tgt
        )


class STShotSampler(ShotSampler):

    def __init__(self, src_path, tgt_path, src_lang, tgt_lang, max_src_len=None, max_tgt_len=None):
        self.shots = read_shot_paths(src_path, tgt_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
        self._template = "Speech: {example_src}\n {tgt_lang}: {example_tgt}"
        self._tgt_lang = tgt_lang

    def sample_template(self):
        example_src, example_tgt = self.sample()
        return self.template.format(
            example_src=example_src,
            example_tgt=example_tgt,
            tgt_lang=self._tgt_lang
        )


def prep_example(example, src_lang, tgt_lang, template, n_shots=0, shot_sampler=None, tokenizer_for_chat=None, gold_transcription=None):

    assert n_shots == 0 or shot_sampler is not None

    template_args = {"example": example}
    if src_lang is not None:
        template_args["src_lang"] = src_lang
    if tgt_lang is not None:
        template_args["tgt_lang"] = tgt_lang
    if gold_transcription is not None:
        template_args["gold_transcription"] = gold_transcription
    prompt = template.format(**template_args)

    if n_shots > 0:
        shots = shot_sampler.sample_context(n=n_shots)
        prompt = shots + prompt

    if tokenizer_for_chat is not None and tokenizer_for_chat.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer_for_chat.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return prompt


def prep_example_multiturn(example, gold_transcription, src_lang, tgt_lang, template, tokenizer):
    """
    The gold_transcription does not need to be gold, I'm just keeping the
    variable name for convenience.
    """

    template_args = {"example": example}
    if src_lang is not None:
        template_args["src_lang"] = src_lang
    if tgt_lang is not None:
        template_args["tgt_lang"] = tgt_lang
    if gold_transcription is not None:
        template_args["transcription"] = gold_transcription

    messages = []
    for i, turn in enumerate(template):
        turn_taker = "user" if i % 2 == 0 else "assistant"
        content = turn.format(**template_args)
        messages.append({"role": turn_taker, "content": content})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


def main(args):
    templates = {
        "mt_zero": "Translate the following text from {src_lang} to {tgt_lang}.\n{src_lang}: {example}\n{tgt_lang}:",
        "mt_icl": "{src_lang}: {example}\n{tgt_lang}:",
        "asr_simple": "Speech: {example}\n English:",
        "asr_nolang": "Speech: {example}\n Text:",
        "asr_bothlang": "English speech: {example}\n English text:",
        "asr_spiritlm": "[Speech] {example} \n[Text]",
        "st_simple": "Speech: {example}\n {tgt_lang}:"
    }
    assert args.template in templates
    assert (args.shot_src is None) == (args.shot_tgt is None)
    assert args.n_shots == 0 or args.shot_src is not None
    random.seed(a=args.seed)

    if args.shot_src is not None:
        if args.template.startswith("st"):
            sampler = STShotSampler(args.shot_src, args.shot_tgt, args.src_lang, args.tgt_lang)
        elif args.template.startswith("mt"):
            sampler = ShotSampler(args.shot_src, args.shot_tgt, args.src_lang, args.tgt_lang)
        else:
            asr_model = "tower" if args.template != "asr_spiritlm" else "spiritlm"
            sampler = ASRShotSampler(
                args.shot_src,
                args.shot_tgt,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                model=asr_model
            )
    else:
        sampler = None

    if args.chat_tokenizer is not None:
        chat_tokenizer = AutoTokenizer.from_pretrained(args.chat_tokenizer)
    else:
        chat_tokenizer = None

    template = templates[args.template]
    if isinstance(template, list):
        prep_ex = partial(
            prep_example_multiturn,
            template=template,
            tokenizer=chat_tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
    else:
        prep_ex = partial(
            prep_example,
            template=template,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            n_shots=args.n_shots,
            shot_sampler=sampler,
            tokenizer_for_chat=chat_tokenizer
        )

    if args.src_extra is not None:
        with open(args.src) as srcf, open(args.src_extra) as extraf:
            instructions = [{"instruction": prep_ex(line.strip(), gold_transcription=extra_line.strip())}
                            for line, extra_line in zip(srcf, extraf)]
    else:
        with open(args.src) as srcf:
            instructions = [{"instruction": prep_ex(line.strip())} for line in srcf]

    with open(args.out, "w") as outf:
        json.dump(instructions, outf, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to source sequences (plain text)")
    parser.add_argument("--src-extra", default=None)
    parser.add_argument("--out", required=True, help="Path to save instructions to")
    parser.add_argument("--n-shots", type=int, default=0)
    parser.add_argument("--shot-src", help="For construction in-context learning examples")
    parser.add_argument("--shot-tgt", help="For construction in-context learning examples")
    parser.add_argument("--max-src-len", type=int, default=None)
    parser.add_argument("--max-tgt-len", type=int, default=None)
    parser.add_argument("--src-lang")
    parser.add_argument("--tgt-lang")
    parser.add_argument("--template", choices=["mt_zero", "mt_icl", "asr_simple", "st_simple"])
    parser.add_argument("--chat-tokenizer", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
