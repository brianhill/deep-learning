import os
import re
import tiktoken
# from importlib.metadata import version
# print("tiktoken version:", version("tiktoken"))  # 0.9.0
import torch
from torch.utils.data import Dataset, DataLoader

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

# Raschka p. 35

enc_text = bpe_tokenizer.encode(raw_text)

print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)
#     print(bpe_tokenizer.decode(context), "---->", bpe_tokenizer.decode([desired]))

# Raschka p. 37

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

dataloader_v1_1 = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)

data_iter = iter(dataloader_v1_1)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader_v1_2 = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader_v1_2)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Raschka p. 42
