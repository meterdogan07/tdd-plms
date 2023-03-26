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
    data_dir1 = "/userfiles/merdogan18/oscar_2301/tr_meta_part_1.jsonl"
    data_dir = "/userfiles/merdogan18/oscar_2301/"
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    dict_count = {}
    avg_dict_count = {}
    total_samples = 0
    total_filesize = 0
    ct_doc = 0
    G_params = [0,0,0,0,0,0,0,0,0,0,0]

    formatted_dataset = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-newoscar/newoscar_data_formatted.json", "a")
    count_dictionaries = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-newoscar/newoscar_stats_formatted.json", "a")
    
    for doc_num in range(1,51):
        print(doc_num)
        dict_count[str(doc_num)] = {}
        avg_dict_count[str(doc_num)] = {}
        dir_name = data_dir+"tr_meta_part_"+str(doc_num)+".jsonl"
        data_ins = [json.loads(line) for line in open(dir_name, 'r', encoding='utf-8')]
        dict_count[str(doc_num)]["sample_size"] = len(data_ins)
        avg_dict_count[str(doc_num)]["sample_size"] = len(data_ins)
        total_samples += len(data_ins)
        total_filesize += os.path.getsize(dir_name) 
        params = [0,0,0,0,0,0,0,0,0,0,0]
        params_array = np.zeros((11,len(data_ins)))
        
        for i in range(len(data_ins)): #range(len(data_ins)):
            txt = data_ins[i]['content']

            dict_sample = {}
            dict_sample["source"] = data_ins[1]['warc_headers']['warc-target-uri']
            dict_sample["doc_id"] = ct_doc
            dict_sample["language"] = "tr"
            dict_sample["corpus"] = "oscar2301"
            dict_sample["context"] = "null"
            dict_sample["text"] = txt
            ct_doc += 1

            json.dump(dict_sample, formatted_dataset, ensure_ascii=False)
            formatted_dataset.write("\n")
            
            params_array[0][i] = len(nltk.word_tokenize(txt, language='turkish'))
            params_array[1][i] = len(nltk.sent_tokenize(txt, language='turkish'))
            params_array[2][i] = count(txt, string.punctuation)
            params_array[3][i] = count(txt, string.ascii_letters)
            params_array[4][i] = count(txt, string.ascii_lowercase)
            params_array[5][i] = count(txt, string.ascii_uppercase)
            params_array[6][i] = count(txt, string.digits)
            params_array[7][i] = count(txt, string.whitespace)
            params_array[8][i] = len(txt.split())
            params_array[9][i] = len(txt.split("\n"))
            params_array[10][i] = len(txt)

            params[0] += params_array[0][i]
            params[1] += params_array[1][i]
            params[2] += params_array[2][i]
            params[3] += params_array[3][i]
            params[4] += params_array[4][i]
            params[5] += params_array[5][i]
            params[6] += params_array[6][i]
            params[7] += params_array[7][i]
            params[8] += params_array[8][i]
            params[9] += params_array[9][i]
            params[10] += params_array[10][i]

            G_params[0] += params_array[0][i]
            G_params[1] += params_array[1][i]
            G_params[2] += params_array[2][i]
            G_params[3] += params_array[3][i]
            G_params[4] += params_array[4][i]
            G_params[5] += params_array[5][i]
            G_params[6] += params_array[6][i]
            G_params[7] += params_array[7][i]
            G_params[8] += params_array[8][i]
            G_params[9] += params_array[9][i]
            G_params[10] += params_array[10][i]

        dict_count[str(doc_num)]["len"] = params_array[10].tolist()
        dict_count[str(doc_num)]["nltk_word"] = params_array[0].tolist()
        dict_count[str(doc_num)]["nltk_sentence"] = params_array[1].tolist()
        dict_count[str(doc_num)]["punctutation"] = params_array[2].tolist()
        dict_count[str(doc_num)]["ascii_letters"] = params_array[3].tolist()
        dict_count[str(doc_num)]["ascii_lowercase"] = params_array[4].tolist()
        dict_count[str(doc_num)]["ascii_uppercase"] = params_array[5].tolist()
        dict_count[str(doc_num)]["digits"] = params_array[6].tolist()
        dict_count[str(doc_num)]["whitespace"] = params_array[7].tolist()
        dict_count[str(doc_num)]["word"] = params_array[8].tolist()
        dict_count[str(doc_num)]["line"] = params_array[9].tolist()

        json.dump(dict_count,count_dictionaries, ensure_ascii=False)
        count_dictionaries.write("\n")

        dict_count = {}
        
        """avg_dict_count[str(doc_num)]["size(GB)_total"] = os.path.getsize(dir_name)/(1024*1024*1024)

        avg_dict_count[str(doc_num)]["length_total"] = params[10]
        avg_dict_count[str(doc_num)]["nltk_word_total"] = params[0]
        avg_dict_count[str(doc_num)]["nltk_sentence_total"] = params[1]
        avg_dict_count[str(doc_num)]["punctutation_total"] = params[2]
        avg_dict_count[str(doc_num)]["ascii_letters_total"] = params[3]
        avg_dict_count[str(doc_num)]["ascii_lowercase_total"] = params[4]
        avg_dict_count[str(doc_num)]["ascii_uppercase_total"] = params[5]
        avg_dict_count[str(doc_num)]["digits_total"] = params[6]
        avg_dict_count[str(doc_num)]["whitespace_total"] = params[7]
        avg_dict_count[str(doc_num)]["word_total"] = params[8]
        avg_dict_count[str(doc_num)]["line_total"] = params[9]

        avg_dict_count[str(doc_num)]["length_avg"] = params[10]/len(data_ins)
        avg_dict_count[str(doc_num)]["nltk_word_avg"] = params[0]/len(data_ins)
        avg_dict_count[str(doc_num)]["nltk_sentence_avg"] = params[1]/len(data_ins)
        avg_dict_count[str(doc_num)]["punctutation_avg"] = params[2]/len(data_ins)
        avg_dict_count[str(doc_num)]["ascii_letters_avg"] = params[3]/len(data_ins)
        avg_dict_count[str(doc_num)]["ascii_lowercase_avg"] = params[4]/len(data_ins)
        avg_dict_count[str(doc_num)]["ascii_uppercase_avg"] = params[5]/len(data_ins)
        avg_dict_count[str(doc_num)]["digits_avg"] = params[6]/len(data_ins)
        avg_dict_count[str(doc_num)]["whitespace_avg"] = params[7]/len(data_ins)
        avg_dict_count[str(doc_num)]["word_avg"] = params[8]/len(data_ins)
        avg_dict_count[str(doc_num)]["line_avg"] = params[9]/len(data_ins)"""

    avg_dict_count["all"] = {}
    avg_dict_count["all"]["size(GB)"] = total_filesize/(1024*1024*1024)
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

    total_json_object = json.dumps(avg_dict_count, indent=20, ensure_ascii=False)
    with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-newoscar/summary_newoscar_stats_formatted.json", "w") as outfile:
        outfile.write(total_json_object)

    formatted_dataset.close()
    count_dictionaries.close()

if __name__ == "__main__":
    main()
