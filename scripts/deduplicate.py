"""
Expects tokens in PUA format
"""

import sys
from spire.utils import deduplicate

for line in sys.stdin:
    sys.stdout.write("".join(deduplicate(line)))
