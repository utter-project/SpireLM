import argparse
from os.path import join

import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


def load_tsv(path):
    """
    Return a list of wav files (as I understand it, the lengths should not be
    relevant for this step)
    """
    with open(path) as f:
        parent_dir = f.readline().strip()  # should just be root
        return [join(parent_dir, line.strip().split("\t")[0]) for line in f]


def generate(prompts, args):
    # dataset = load_dataset("openslr/librispeech_asr", "clean",split='test')["audio"]
    pipe = pipeline(
        "automatic-speech-recognition",
        args.model,
        chunk_length_s=args.chunk_len,
        batch_size=args.batch_size,
        torch_dtype=args.dtype,
        device=args.device,
    )
    predictions = []
    for i in tqdm.trange(0, len(prompts), args.batch_size):
        batch = prompts[i: i + args.batch_size]
        preds = pipe(batch, max_new_tokens=args.max_length, generate_kwargs={"language": "english"})
        for pred in preds:
            predictions.append(pred["text"])

    return predictions


def main(args):
    wavs = load_tsv(args.input_tsv)
    predictions = generate(wavs, args)

    with open(args.outpath, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--tokenizer", default="openai/whisper-large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0", help="Only used with backend=hf")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk-len", type=int, default=30)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--input-tsv")
    parser.add_argument("--outpath")
    args = parser.parse_args()
    main(args)
