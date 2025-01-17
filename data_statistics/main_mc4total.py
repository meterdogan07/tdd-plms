import os
import argparse
import pandas as pd
import json
import string
from multiprocessing import cpu_count
from datasets import load_dataset
import numpy as np

#from filtering import DatasetFiltering

def main():
    
    from_to = 128 # how many files to work on
    dict_count = {}
    avg_dict_count = {}
    total_samples = 0
    G_params = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    for i in range(8):
        start_ix = i*128 # index of the first file
        str_from_to = "_"+str(start_ix)+"to"+str(start_ix+from_to)+"_"
        # with open("/kuacc/users/merdogan18/tdd-plms/preprocessing/oscar/summary_xmc4__0to128_stats.json", 'r') as openfile:
        with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-MC4/summary_mc4_"+str_from_to+"stats_formatted.json", 'r') as openfile:
            json_object = json.load(openfile)

            G_params[0] += json_object["all"]["punctutation_total"]
            G_params[1] += json_object["all"]["ascii_letters_total"]
            G_params[2] += json_object["all"]["ascii_lowercase_total"]
            G_params[3] += json_object["all"]["ascii_uppercase_total"]
            G_params[4] += json_object["all"]["digits_total"]
            G_params[5] += json_object["all"]["size(GB)"]
            G_params[6] += json_object["all"]["whitespace_total"]
            G_params[7] += json_object["all"]["word_total"]
            G_params[8] += json_object["all"]["line_total"]
            G_params[9] += json_object["all"]["total_samples"]
            G_params[10] += json_object["all"]["nltk_word_total"]
            G_params[11] += json_object["all"]["nltk_sentence_total"]
            G_params[12] += json_object["all"]["length_total"]

    total_samples = G_params[9]
    avg_dict_count["all"] = {}
    avg_dict_count["all"]["size(GB)"] = G_params[5]
    avg_dict_count["all"]["total_samples"] = G_params[9]
    avg_dict_count["all"]["length_total"] = G_params[12]
    avg_dict_count["all"]["nltk_word_total"] = G_params[10]
    avg_dict_count["all"]["nltk_sentence_total"] = G_params[11]
    avg_dict_count["all"]["punctutation_total"] = G_params[0]
    avg_dict_count["all"]["ascii_letters_total"] = G_params[1]
    avg_dict_count["all"]["ascii_lowercase_total"] = G_params[2]
    avg_dict_count["all"]["ascii_uppercase_total"] = G_params[3]
    avg_dict_count["all"]["digits_total"] = G_params[4]
    avg_dict_count["all"]["whitespace_total"] = G_params[6]
    avg_dict_count["all"]["word_total"] = G_params[7]
    avg_dict_count["all"]["line_total"] = G_params[8]

    avg_dict_count["all"]["length_avg"] = G_params[12]/total_samples
    avg_dict_count["all"]["nltk_word_avg"] = G_params[10]/total_samples
    avg_dict_count["all"]["nltk_sentence_avg"] = G_params[11]/total_samples
    avg_dict_count["all"]["punctutation_avg"] = G_params[0]/total_samples
    avg_dict_count["all"]["ascii_letters_avg"] = G_params[1]/total_samples
    avg_dict_count["all"]["ascii_lowercase_avg"] = G_params[2]/total_samples
    avg_dict_count["all"]["ascii_uppercase_avg"] = G_params[3]/total_samples
    avg_dict_count["all"]["digits_avg"] = G_params[4]/total_samples
    avg_dict_count["all"]["whitespace_avg"] = G_params[6]/total_samples
    avg_dict_count["all"]["word_avg"] = G_params[7]/total_samples
    avg_dict_count["all"]["line_avg"] = G_params[8]/total_samples

    total_json_object = json.dumps(avg_dict_count, indent=2)
    with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-MC4/summary_mc4_total_stats_formatted.json", "w") as outfile:
        outfile.write(total_json_object)

if __name__ == "__main__":
    main()
                      