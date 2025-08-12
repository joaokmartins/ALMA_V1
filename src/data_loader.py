import re

# Carregar o arquivo de texto do local onde foi salvo
with open("data/the-verdict.txt", "r", encoding="utf-8") as f: 
    # <--- IMPORTANTE: Ler da pasta 'data'
    raw_text = f.read()
print("Total number of character:", len(raw_text)) 
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s+)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])   

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

print(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break    

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab #1
        self.int_to_str = {i:s for s,i in vocab.items()} #2

    def encode(self, text): #3
        preprocessed = re.split(r'([,.?_!"()\']|--|\s+)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
            ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids): #4
        text = " ".join([self.int_to_str[i] for i in ids])
        
        text = re.sub(r"\s+([,.?!\"()\\'])", r"\1", text) #5
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))