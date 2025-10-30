import re
from simple_tokenizer import SimpleTokenizerV1

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
result = re.split(r'(\s)', raw_text)

print(result)

# tokenizer = SimpleTokenizerV1(vocab)

# text = """"It's the last he painted, you know," 
#            Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)