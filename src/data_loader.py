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

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item) 

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int             #1
                        else "<|unk|>" for item in preprocessed]
      
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
   
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))