import json
import sys

def bad_example(line):
    example = json.loads(line)
    turns = example["conversations"]
    return any(not turn["value"] for turn in turns)


if __name__ == "__main__":
    for line in sys.stdin:
        if not bad_example(line):
            sys.stdout.write(line)
