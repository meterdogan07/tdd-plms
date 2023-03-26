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
    dir2 = "/userfiles/merdogan18/data_statistics/Processed-SketchEngine/sketchengine_data_formatted.json"
    #formatted_dataset = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-SketchEngine/sketchengine_data_formatted22xx.json", "a")

    ct2 = 0
    for line in open(dir2, 'r', encoding='utf-8'):
        print(dir2)
        ct2 += 1
        linell = json.loads(line)
        print(ct2, " ", linell['doc_id'])
        #print(linell)
        #print()
        """
        if('doc_id' in linell.keys()):
            #json.dump(linell,formatted_dataset, ensure_ascii=False)
            #formatted_dataset.write("\n")
            print(ct2, " ", linell['doc_id'])
        else:
            print(linell)
        """
    print("end of sketchengine")

if __name__ == "__main__":
    main()
