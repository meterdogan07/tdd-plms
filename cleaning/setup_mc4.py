from datasets import load_dataset, load_from_disk
from datasets import Dataset
from itertools import product
import logging
import os
from collections import Counter, defaultdict, deque
from typing import Dict, Set

import gcsfs
import simhash
import typer
import yaml
import datasets
from datasets import load_dataset, DatasetDict
from datasets.load import load_from_disk
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm

output_file = open("./ProperMC4/mc4.txt", "w")

conf = "./deduplicate/conf/self_deduplicate_tr.yaml"

with open(conf, "r") as f:
    conf = yaml.safe_load(f.read())

if conf["load_from_disk"]["path"]:
    fs: AbstractFileSystem = None
    if conf["load_from_disk"]["gcs"]:
        fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
    ds = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
else:
    ds = load_dataset(**conf["load_dataset"])
    
def remove_whitespace(sentence):
    return sentence.replace("\n", "").replace("\r", "").replace("\\n", "")
    
for i in tqdm(range(len(ds)), total=len(ds)):
    output_file.write(remove_whitespace(ds[i]['text']) + "\n")
    
output_file.close()
