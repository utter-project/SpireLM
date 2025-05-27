import sys
from spire.utils import deduplicate

for line in sys.stdin:
    line = line.strip()
    out_line = "".join(deduplicate(list(line)))
    sys.stdout.write(out_line + "\n")
