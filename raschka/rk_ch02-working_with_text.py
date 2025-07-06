import os
import re

# Raschka p. 22

LLMS_FROM_SCRATCH_BASE = "../../llms-from-scratch"
THE_VERDICT_FILENAME = "ch02/01_main-chapter-code/the-verdict.txt"

the_verdict_full_path = os.path.join(LLMS_FROM_SCRATCH_BASE, THE_VERDICT_FILENAME)

with open(the_verdict_full_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

# Raschka p. 23
