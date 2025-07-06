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

newResult = re.split(r'([,.\s])', text)
print(newResult)

newText = "Hello, world. Is this-- a test?"
newNewResult = [item.strip() for item in re.split(r'([,.]|--|\s)', newText)
                if len(item.strip()) > 0]
print(newNewResult)

# Raschka p. 24

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# Raschka p. 25

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# Raschka p. 27
