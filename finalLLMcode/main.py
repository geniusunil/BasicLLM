# import importlib
import re
import torch
import tiktoken
from simple_tokenizer import SimpleTokenizerV2
from gpt_dataset_v1 import GPTDatasetV1
# import importlib
# print(importlib.__file__)

# print("tiktoken version:", importlib.metadata.version("tiktoken"))

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# text = raw_text.join(" <|endoftext|> ")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
# # result = re.split(r'(\s)', raw_text)
all_words.extend(["<|endoftext|>", "<|unk|>"])
# # print(result)
vocab = {token:integer for integer,token in enumerate(all_words)}

# tokenizer = SimpleTokenizerV2(vocab)

# # text = """"It's the last he painted, you know," 
# #            Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(raw_text + " <|endoftext|> ")
# print(ids)

# tokenizer = tiktoken.get_encoding("gpt2")

# integers = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
# gptdataset = GPTDatasetV1(raw_text)
# print("PyTorch version:", torch.__version__)
dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
# print(integers)

# strings = tokenizer.decode(integers)

# print(strings)
# input_ids = torch.tensor([2, 3, 5, 1])
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# print(embedding_layer.weight)
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

code = tokenizer.encode(text)

print(code)

print(tokenizer.decode(code))