import os
import re
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# Raschka p. 22

LLMS_FROM_SCRATCH_BASE = "../../llms-from-scratch"
THE_VERDICT_FILENAME = "ch02/01_main-chapter-code/the-verdict.txt"

the_verdict_full_path = os.path.join(LLMS_FROM_SCRATCH_BASE, THE_VERDICT_FILENAME)

with open(the_verdict_full_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    # print("Total number of characters:", len(raw_text))
    # print(raw_text[:99])

testText = "Hello, world. This, is a test."
testResult = re.split(r'(\s)', testText)
# print(testResult)

# Raschka p. 23

newResult = re.split(r'([,.\s])', testText)
# print(newResult)

newTestText = "Hello, world. Is this-- a test?"
newNewResult = [item.strip() for item in re.split(r'([,.]|--|\s)', newTestText) if item.strip()]
# print(newNewResult)

# Raschka p. 24

testPreprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
testPreprocessed = [item.strip() for item in testPreprocessed if item.strip()]
# print(len(testPreprocessed))
# print(testPreprocessed[:30])

# Raschka p. 25

all_words = sorted(set(testPreprocessed))
# print(len(all_words))

testVocab = {token: integer for integer, token in enumerate(all_words)}

# for ii, item in enumerate(testVocab.items()):
#     print(item)
#     if ii >= 50:
#         break

# Raschka p. 27

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizerV1 = SimpleTokenizerV1(testVocab)
moreTestText = """
"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
"""

testIds = tokenizerV1.encode(moreTestText)
print(testIds)
# A couple of stray spaces are present, but they also appear on p. 28.
print(tokenizerV1.decode(testIds))

# Raschka p. 30

all_tokens = sorted(list(set(testPreprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
extended_vocab = {token: integer for integer, token in enumerate(all_tokens)}

# print(len(extended_vocab.items()))

# for ii, item in enumerate(list(extended_vocab.items())[-5:]):
#     print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text1_joined_with_text2 = " <|endoftext|> ".join((text1, text2))

print(text1_joined_with_text2)

tokenizerV2 = SimpleTokenizerV2(extended_vocab)

print(tokenizerV2.decode(tokenizerV2.encode(text1_joined_with_text2)))

# Raschka p. 33

bpe_tokenizer = tiktoken.get_encoding("gpt2")

tik_test_text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someUnknownPlace."
)
tik_test_integers = bpe_tokenizer.encode(tik_test_text, allowed_special={"<|endoftext|>"})
print(tik_test_integers)

tik_test_strings = bpe_tokenizer.decode(tik_test_integers)
print(tik_test_strings)

# NEED TO READ p. 34 BEFORE GOING ON TO p. 35 (THE END OF THE CHAPTER IS p. 49)
