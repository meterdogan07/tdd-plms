import json

filename256  = "/kuacc/users/asafaya19/tdd-corpora/tdd-corpus-00-256.txt"
filename1024 = "/kuacc/users/asafaya19/tdd-corpora/tdd-corpus-00-1024.txt"

l256, l1024 = [], []
with open(filename256, "r") as f:
    lines = f.readlines()
    for l in lines:
        l = l.rstrip("\n")
        l256.append(l)

with open(filename1024, "r") as f:
    lines = f.readlines()
    for l in lines:
        l = l.rstrip("\n")
        l1024.append(l)

json_lines_256  = [json.dumps(l, ensure_ascii=False) for l in l256 ]
json_lines_1024 = [json.dumps(l, ensure_ascii=False) for l in l1024]

# Join lines and save to .jsonl file
json_data = '\n'.join(json_lines_256)
with open('tdd-corpus-00-256.jsonl', 'w') as f:
    f.write(json_data)

json_data = '\n'.join(json_lines_1024)
with open('tdd-corpus-00-1024.jsonl', 'w') as f:
    f.write(json_data)
