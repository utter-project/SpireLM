"""
Simple, straightforward offline inference with vllm.
"""

import argparse
import json

from vllm import LLM, SamplingParams


def generate(prompts, model, args):
    stops = ["<\s>", '\\n', '\n', "<END>"]

    sampling_args = {
        "best_of": 1,
        "n": 1,
        "temperature": args.temperature,
        "max_tokens": args.max_length,
        "stop": stops
    }

    sampling_params = SamplingParams(**sampling_args)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = model.generate(prompts, sampling_params)
    predictions = [output.outputs[0].text for output in outputs]

    return predictions


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line)["instruction"] for line in f]


def main(args):
    assert len(args.inpaths) == len(args.outpaths)

    # hints for why results may not align with HF:
    # https://github.com/vllm-project/vllm/pull/1885
    tokenizer_path = args.tokenizer if args.tokenizer is not None else args.model

    model = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        tokenizer_mode="slow" if args.slow_tokenizer else "auto",
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size
    )

    for inpath, outpath in zip(args.inpaths, args.outpaths):
        prompts = load_jsonl(inpath)

        predictions = generate(prompts, model, args)

        with open(outpath, "w") as f:
            for prediction in predictions:
                f.write(prediction + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None, help="Specify if different from args.model")
    parser.add_argument("--slow-tokenizer", action="store_true", help="Necessary for properly handling <extra_id_{}>")
    parser.add_argument("--inpaths", nargs="+")
    parser.add_argument("--outpaths", nargs="+")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="you probably shouldn't touch this")
    args = parser.parse_args()
    main(args)
