import json

filename="/kuacc/users/eacikgoz17/tdd/tdd-plms/tokenization/Tokenizers/BPE.byte.tokenizer.48k.json"

with open(filename, "r") as f:
    data   = json.load(f)
    merges = data["model"]["merges"]
    vocab  = data["model"]["vocab"]


with open("merges.txt", "w") as f:
    for merge in merges:
        f.write(merge + "\n")

with open('vocab.json', 'w', encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)


