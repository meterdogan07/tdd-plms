import os
import argparse
import pandas as pd
import json
import string
import codecs
import nltk
import nltk.tokenize
from multiprocessing import cpu_count
from datasets import load_dataset
import numpy as np

def main():
    dir1 = "/userfiles/merdogan18/data_statistics/Processed-MC4/mc4__0to128_data_formatted.json"
    dir2 = "/userfiles/merdogan18/tdd-plms/data_statistics/Processed-Oscar/oscar_data_formatted.json"
    dir3 = "/userfiles/merdogan18/data_statistics/Processed-SketchEngine/sketchengine_data_formatted.json"
    formatted_dataset = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/customdata.json", "a")

    ct1 = 0
    for line in open(dir1, 'r', encoding='utf-8'):
        if(ct1 >= 600000):
            break
        linell = json.loads(line)
        linell['doc_id'] = ct1
        ct1 += 1
        json.dump(linell, formatted_dataset, ensure_ascii=False)
        formatted_dataset.write("\n")
        

    for line in open(dir2, 'r', encoding='utf-8'):
        if(ct1 >= 700000):
            break
        linell = json.loads(line)
        linell['doc_id'] = ct1
        ct1 += 1
        json.dump(linell, formatted_dataset, ensure_ascii=False)
        formatted_dataset.write("\n")
        

    for line in open(dir3, 'r', encoding='utf-8'):
        if(ct1 >= 700000):
            break
        linell = json.loads(line)
        linell['doc_id'] = ct1
        ct1 += 1
        json.dump(linell, formatted_dataset, ensure_ascii=False)
        formatted_dataset.write("\n")
        

    formatted_dataset.close()
if __name__ == "__main__":
    main()
