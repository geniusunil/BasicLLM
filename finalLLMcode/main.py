# import importlib
import re
import tiktoken
from simple_tokenizer import SimpleTokenizerV2

# import importlib
# print(importlib.__file__)

# print("tiktoken version:", importlib.metadata.version("tiktoken"))

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# text = raw_text.join(" <|endoftext|> ")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
# result = re.split(r'(\s)', raw_text)
all_words.extend(["<|endoftext|>", "<|unk|>"])
# print(result)
vocab = {token:integer for integer,token in enumerate(all_words)}

tokenizer = SimpleTokenizerV2(vocab)

# text = """"It's the last he painted, you know," 
#            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(raw_text + " <|endoftext|> ")
print(ids)