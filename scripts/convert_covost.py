import argparse
import subprocess  # maybe we move this to a different script
import os
from os.path import basename, join, splitext
from tqdm import tqdm
import torchaudio


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.audio_list) as inp_f:
        mp3_files = [join(args.audio_dir, line.strip()) for line in inp_f]

    with open(args.output_tsv, "w") as f:
        f.write("/\n")
        for mp3_file in tqdm(mp3_files):
            wav_path = join(args.output_dir, splitext(basename(mp3_file))[0] + ".wav")
            subprocess.run(["ffmpeg", "-i", mp3_file, "-ar", "16000", wav_path])
            length = torchaudio.load(wav_path)[0].shape[1]
            f.write("\t".join([wav_path, str(length)]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-list", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-tsv", required=True)
    args = parser.parse_args()
    main(args)
