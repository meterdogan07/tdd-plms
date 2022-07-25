import sys
import os
import json
import random

from collections import Counter
from tqdm import tqdm

from glob import glob
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, Regex
from tokenizers import processors
from tokenizers.models import Unigram

unigram_models = [f"Unigram.spiece.tokenizer.trwiki-67.{size}k.json" for size in ["32", "64"]]
bpe_models = [f"BPE.spiece.tokenizer.trwiki-67.{size}k.json" for size in ["32", "64"]]

text_files = [f"tdd0{x}.txt" for x in range(10)]

log = open("log.txt", "w")

if not os.path.isdir("Frequencies"):
    os.mkdir("Frequencies")

def read_file(file):
    with open(file, "r") as f:
        return f.readlines()
    
    
def get_frequencies(model):
    
    text_counts = Counter()
    
    log.write("Tokenizer file loaded\n")
    
    tokenizer = Tokenizer.from_file(f"Tokenizers/{model}")
    
    log.write("Starting to load text files\n")
    for file in tqdm(text_files, total=10):
        text = read_file(file)

        encoded = tokenizer.encode_batch(text)
        indices = [line.ids for line in encoded]
        for index in indices:
            text_counts.update(index)
        
    return text_counts
    
with open("Frequencies/Unigram_32k.txt", "w") as f:    
    counter = get_frequencies(model=unigram_models[0])
    for k, v in counter.most_common():
        f.write(f"{k}, {v}\n")
with open("Frequencies/Unigram_64k.txt", "w") as f:    
    counter = get_frequencies(model=unigram_models[1])
    for k, v in counter.most_common():
        f.write(f"{k}, {v}\n")
with open("Frequencies/BPE_32k.txt", "w") as f:    
    counter = get_frequencies(model=bpe_models[0])
    for k, v in counter.most_common():
        f.write(f"{k}, {v}\n")
with open("Frequencies/BPE_64k.txt", "w") as f:    
    counter = get_frequencies(model=bpe_models[1])
    for k, v in counter.most_common():
        f.write(f"{k}, {v}\n")