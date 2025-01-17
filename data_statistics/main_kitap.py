import os
import argparse
import pandas as pd
import json
import string
from multiprocessing import cpu_count
from datasets import load_dataset
import nltk
import nltk.tokenize
import numpy as np

def main():
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    data_dir = "/userfiles/merdogan18/EngindenKitap/"
    directories = ["A-C","C-E","F-H","I-K","S-T","U-Z"] #,"C-E","F-H","I-K","S-T","U-Z"
    
    dict_count = {}
    avg_dict_count = {}
    total_samples = 0
    G_params = [0,0,0,0,0,0,0,0,0,0,0]
    total_size = 0
    
    for dir_num in directories:
        path = data_dir+dir_num+"/"
        dict_count[dir_num] = {}
        dir_list = os.listdir(path)
        
        for file in dir_list:
            file_dir = path+file
            if(file_dir[-4:] == ".txt" and file_dir[-8:-4] != ".txt" ):
                total_samples += 1
                total_size += os.path.getsize(file_dir)
                f = open(file_dir, "r", encoding='utf-8')
                txt = f.read()
                f.close()
                print(file_dir)

                G_params[0] += len(nltk.word_tokenize(txt, language='turkish'))
                G_params[1] += len(nltk.sent_tokenize(txt, language='turkish'))
                G_params[2] += count(txt, string.punctuation)
                G_params[3] += count(txt, string.ascii_letters)
                G_params[4] += count(txt, string.ascii_lowercase)
                G_params[5] += count(txt, string.ascii_uppercase)
                G_params[6] += count(txt, string.digits)
                G_params[7] += count(txt, string.whitespace)
                G_params[8] += len(txt.split())
                G_params[9] += len(txt.split("\n"))
                G_params[10] += len(txt)

                dict_count[dir_num][file] = {}
                dict_count[dir_num][file]["nltk_word"] = len(txt)
                dict_count[dir_num][file]["nltk_word"] = len(nltk.word_tokenize(txt, language='turkish'))
                dict_count[dir_num][file]["nltk_sentence"] = len(nltk.sent_tokenize(txt, language='turkish'))
                dict_count[dir_num][file]["punctutation"] = count(txt, string.punctuation)
                dict_count[dir_num][file]["ascii_letters"] = count(txt, string.ascii_letters)
                dict_count[dir_num][file]["ascii_lowercase"] = count(txt, string.ascii_lowercase)
                dict_count[dir_num][file]["ascii_uppercase"] = count(txt, string.ascii_uppercase)
                dict_count[dir_num][file]["digits"] = count(txt, string.digits)
                dict_count[dir_num][file]["whitespace"] = count(txt, string.whitespace)
                dict_count[dir_num][file]["word"] = len(txt.split())
                dict_count[dir_num][file]["line"] = len(txt.split("\n"))

        json_object = json.dumps(dict_count, indent=1)
        with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/preprocessing/oscar/Processed-Kitap/Engindenkitap_stats.json", "w") as outfile:
            outfile.write(json_object)
        dict_count = {}

    avg_dict_count["all"] = {}
    avg_dict_count["all"]["size(GB)"] = total_size/(1024*1024*1024)
    avg_dict_count["all"]["total_samples"] = total_samples

    avg_dict_count["all"]["length_total"] = G_params[10]
    avg_dict_count["all"]["nltk_word_total"] = G_params[0]
    avg_dict_count["all"]["nltk_sentence_total"] = G_params[1]
    avg_dict_count["all"]["punctutation_total"] = G_params[2]
    avg_dict_count["all"]["ascii_letters_total"] = G_params[3]
    avg_dict_count["all"]["ascii_lowercase_total"] = G_params[4]
    avg_dict_count["all"]["ascii_uppercase_total"] = G_params[5]
    avg_dict_count["all"]["digits_total"] = G_params[6]
    avg_dict_count["all"]["whitespace_total"] = G_params[7]
    avg_dict_count["all"]["word_total"] = G_params[8]
    avg_dict_count["all"]["line_total"] = G_params[9]

    avg_dict_count["all"]["length_avg"] = G_params[10]/total_samples
    avg_dict_count["all"]["nltk_word_avg"] = G_params[0]/total_samples
    avg_dict_count["all"]["nltk_sentence_avg"] = G_params[1]/total_samples
    avg_dict_count["all"]["punctutation_avg"] = G_params[2]/total_samples
    avg_dict_count["all"]["ascii_letters_avg"] = G_params[3]/total_samples
    avg_dict_count["all"]["ascii_lowercase_avg"] = G_params[4]/total_samples
    avg_dict_count["all"]["ascii_uppercase_avg"] = G_params[5]/total_samples
    avg_dict_count["all"]["digits_avg"] = G_params[6]/total_samples
    avg_dict_count["all"]["whitespace_avg"] = G_params[7]/total_samples
    avg_dict_count["all"]["word_avg"] = G_params[8]/total_samples
    avg_dict_count["all"]["line_avg"] = G_params[9]/total_samples

    # Writing to sample.json
    total_json_object = json.dumps(avg_dict_count, indent=2)
    with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/preprocessing/oscar/Processed-Kitap/summary_Engindenkitap_stats.json", "w") as outfile:
        outfile.write(total_json_object)

if __name__ == "__main__":
    main()
                     