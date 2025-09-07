import argparse
import json
import sys
from os.path import join

import tqdm
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    SeamlessM4Tv2ForSpeechToText, SeamlessM4Tv2ForTextToText, AutoProcessor
from vllm import LLM, SamplingParams


def _generate_vllm(prompts, model, args):
    stops = ["<\s>"]
    stops.extend(['\\n', '\n', "<END>"])

    sampling_args = {
        "use_beam_search": args.beam_size > 1,
        "best_of": args.beam_size,
        "n": 1,
        "temperature": args.temperature,
        "max_tokens": args.max_length,
        "stop": stops
    }

    if args.beam_size > 1:
        # beam search is deprecated in newer versions of vllm, so beware
        sampling_args["early_stopping"] = args.early_stopping
    sampling_params = SamplingParams(**sampling_args)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = model.generate(prompts, sampling_params)
    predictions = [output.outputs[0].text for output in outputs]

    return predictions


def _generate_hf(prompts, model, args):
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=not args.slow_tokenizer,
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    predictions = []
    for i in tqdm.trange(0, len(prompts), args.batch_size):
        batch = prompts[i: i + args.batch_size]
        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(
                batch,
                padding="longest",
                return_tensors="pt",
                return_token_type_ids=None
            ).to(args.device)
            input_length = inputs.input_ids.shape[1]
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                num_beams=args.beam_size
            )

        decoded_batch = tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)
        predictions.extend([pred.strip() for pred in decoded_batch])

    return predictions


def _generate_hf_seamless(prompts, model, args):

    processor_path = args.tokenizer if args.tokenizer is not None else args.model
    processor = AutoProcessor.from_pretrained(processor_path)

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    predictions = []
    for i in tqdm.trange(0, len(prompts), args.batch_size):
        batch = prompts[i: i + args.batch_size]
        with torch.no_grad():

            if args.seamless_input_type == "text":
                inputs = processor(
                    text=batch, src_lang=args.src_lang, return_tensors="pt"
                ).to(args.device)
            else:
                # is src_lang needed here?
                inputs = processor(
                    audios=batch, src_lang=args.src_lang, return_tensors="pt", sampling_rate=16000
                ).to(args.device)
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                num_beams=args.beam_size,
                tgt_lang=args.tgt_lang
            )

        decoded_batch = processor.batch_decode(output, skip_special_tokens=True)
        predictions.extend([pred.strip() for pred in decoded_batch])

    return predictions


def generate(prompts, model, args):
    if args.backend == "vllm":
        return _generate_vllm(prompts, model, args)

    if args.model_type == "dec":
        return _generate_hf(prompts, model, args)

    # encdec-speech
    return _generate_hf_seamless(prompts, model, args)


def load_json(path):
    with open(path) as f:
        corpus = json.load(f)
        return [c["instruction"] for c in corpus]


def load_wav_tsv(path):
    """
    Return a list of wav files (as I understand it, the lengths should not be
    relevant for this step)
    """
    with open(path) as f:
        parent_dir = f.readline().strip()  # should just be root
        audios = []
        for line in f:
            path = join(parent_dir, line.strip().split("\t")[0])
            example, orig_freq = torchaudio.load(path)
            example = torchaudio.functional.resample(example, orig_freq=orig_freq, new_freq=16_000)
            audios.append(example)
        return audios


def load_raw_text(path):
    with open(path) as f:
        return [line.strip() for line in f]


def main(args):
    assert len(args.inpaths) == len(args.outpaths)

    # hints for why results may not align with HF:
    # https://github.com/vllm-project/vllm/pull/1885
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model

    if args.backend == "vllm":
        model = LLM(
            model=args.model,
            tokenizer=tokenizer_path,
            tokenizer_mode="slow" if args.slow_tokenizer else "auto",
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:
        if args.model_type == "dec":
            model_class = AutoModelForCausalLM
        elif args.seamless_input_type == "speech":
            model_class = SeamlessM4Tv2ForSpeechToText
        else:
            model_class = SeamlessM4Tv2ForTextToText
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}

        model = model_class.from_pretrained(args.model, torch_dtype=dtype[args.dtype]).to(args.device)

    input_readers = {"json": load_json, "wav_tsv": load_wav_tsv, "raw_text": load_raw_text}
    input_reader = input_readers[args.input_format]

    for inpath, outpath in zip(args.inpaths, args.outpaths):
        prompts = input_reader(inpath)

        predictions = generate(prompts, model, args)

        with open(outpath, "w") as f:
            for prediction in predictions:
                f.write(prediction + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.1")
    parser.add_argument("--tokenizer", default=None, help="Specify if different from args.model")
    parser.add_argument("--slow-tokenizer", action="store_true", help="Necessary for properly handling <extra_id_{}> (for DSUs)")
    parser.add_argument("--inpaths", nargs="+")
    parser.add_argument("--outpaths", nargs="+")
    parser.add_argument("--beam-size", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--early-stopping", action="store_true", help="Only relevant for beam search")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--batch-size", default=8, type=int, help="Only used with backend=hf")
    parser.add_argument("--device", default="cuda:0", help="Only used with backend=hf")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="you probably shouldn't touch this")
    parser.add_argument("--src-lang", help="Only used with backend=hf")
    parser.add_argument("--tgt-lang", help="Only used with backend=hf")
    parser.add_argument("--input-format", choices=["json", "wav_tsv", "raw_text"], default="json")
    parser.add_argument("--model-type", choices=["dec", "encdec-speech"],
                        default="dec", help="Only used with backend=hf")
    parser.add_argument("--seamless-input-type", choices=["speech", "text"], default="speech",
                        help="Only for seamless models")
    args = parser.parse_args()
    main(args)
