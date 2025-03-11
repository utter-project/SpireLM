import argparse
import subprocess
import os
from os.path import basename, dirname, join, splitext


def convert_path(native_path, output_dir):
    # last three parts of the path matter
    speaker, chapter = native_path.split("/")[-3:-1]
    filename = splitext(basename(native_path))[0] + ".wav"
    return join(output_dir, speaker, chapter, filename)


def main(args):
    # assume this is only for librispeech
    with open(args.input_tsv) as inp_f, open(args.output_tsv, "w") as out_f:
        header = inp_f.readline()  # assume it's root
        out_f.write(header)
        for line in inp_f:
            native_path, n_samples = line.strip().split("\t")

            if args.to_flac:
                converted_path = join(args.output_dir, splitext(basename(native_path))[0] + ".flac")
            else:
                converted_path = convert_path(native_path, args.output_dir)

            out_f.write("\t".join([converted_path, n_samples]) + "\n")
            os.makedirs(dirname(converted_path), exist_ok=True)

            if args.to_flac:
                subprocess.run(["ffmpeg", "-i", native_path, converted_path])
            else:
                subprocess.run(["ffmpeg", "-i", native_path, "-c:a", "pcm_s24le", converted_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", choices=["ls-clean", "ls-other", "voxpopuli", "fleurs"])
    # parser.add_argument("--path", choices=["google/fleurs", "openslr/librispeech_asr", "facebook/voxpopuli"])
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--to-flac", action="store_true")
    args = parser.parse_args()
    main(args)
