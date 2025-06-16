import argparse


def read_flores(*paths):
    lines = []
    for path in paths:
        with open(path) as f:
            lines.extend([line for line in f])
    return lines


def main(args):
    # for each fleurs src line, find the accompanying line in the target language
    src2target = dict(zip(read_flores(*args.flores_src), read_flores(*args.flores_tgt)))

    with open(args.fleurs_src) as srcf, open(args.fleurs_tgt_inferred, "w") as tgtf:
        for line in srcf:
            assert line in src2target
            tgtf.write(src2target[line])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs-src")
    parser.add_argument("--flores-src", nargs="+")
    parser.add_argument("--flores-tgt", nargs="+")
    parser.add_argument("--fleurs-tgt-inferred")
    args = parser.parse_args()
    main(args)
