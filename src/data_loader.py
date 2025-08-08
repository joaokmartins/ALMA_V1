# Carregar o arquivo de texto do local onde foi salvo
with open("data/the-verdict.txt", "r", encoding="utf-8") as f: 
    # <--- IMPORTANTE: Ler da pasta 'data'
    raw_text = f.read()
print("Total number of character:", len(raw_text)) 
print(raw_text[:99])