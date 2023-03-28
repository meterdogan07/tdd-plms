from multiprocessing import cpu_count
from datasets import load_dataset
import argparse
import numpy as np
import json
import string
import nltk
import nltk.tokenize
import numpy as np


def main():
    #dataset = load_dataset("oscar-corpus/OSCAR-2201", language="tr", use_auth_token=True, split="train", cache_dir="/kuacc/users/asafaya19/turkish-datasets/oscar-2201-tr")
    #dataset = load_dataset("oscar-corpus/OSCAR-2201", language="tr", use_auth_token=True, split="train")
    dataset = load_dataset("oscar-corpus/OSCAR-2201", language="tr", use_auth_token=True, split="train", cache_dir="/kuacc/users/merdogan18/.cache/huggingface/datasets")
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    params = [0,0,0,0,0,0,0,0,0,0,0]
    process_range = len(dataset)
    params_array = np.zeros((11,process_range))

    dict_count = {}
    avg_dict_count = {}

    formatted_dataset = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-Oscar/oscar_data_formatted.json", "a")
    count_dictionaries = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-Oscar/oscar_stats_formatted.json", "a")

    for i in range(process_range):
        print(i)
        txt = dataset[i]['text']

        dict_sample = {}
        dict_sample["source"] = dataset[1]['meta']['warc_headers']['warc-target-uri']
        dict_sample["doc_id"] = i
        dict_sample["language"] = "tr"
        dict_sample["corpus"] = "Oscar"
        dict_sample["context"] = "null"
        dict_sample["text"] = txt

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
    
    
    dict_count["len"] = params_array[10].tolist()
    dict_count["nltk_word"] = params_array[0].tolist()
    dict_count["nltk_sentence"] = params_array[1].tolist()
    dict_count["punctutation"] = params_array[2].tolist()
    dict_count["ascii_letters"] = params_array[3].tolist()
    dict_count["ascii_lowercase"] = params_array[4].tolist()
    dict_count["ascii_uppercase"] = params_array[5].tolist()
    dict_count["digits"] = params_array[6].tolist()
    dict_count["whitespace"] = params_array[7].tolist()
    dict_count["word"] = params_array[8].tolist()
    dict_count["line"] = params_array[9].tolist()


    avg_dict_count["all"] = {}
    avg_dict_count["all"]["size(GB)"] = dataset.info.dataset_size/(1024*1024*1024)
    avg_dict_count["all"]["total_samples"] = len(dataset)

    avg_dict_count["all"]["length_total"] = params[10]
    avg_dict_count["all"]["nltk_word_total"] = params[0]
    avg_dict_count["all"]["nltk_sentence_total"] = params[1]
    avg_dict_count["all"]["punctutation_total"] = params[2]
    avg_dict_count["all"]["ascii_letters_total"] = params[3]
    avg_dict_count["all"]["ascii_lowercase_total"] = params[4]
    avg_dict_count["all"]["ascii_uppercase_total"] = params[5]
    avg_dict_count["all"]["digits_total"] = params[6]
    avg_dict_count["all"]["whitespace_total"] = params[7]
    avg_dict_count["all"]["word_total"] = params[8]
    avg_dict_count["all"]["line_total"] = params[9]

    avg_dict_count["all"]["length_avg"] = params[10]/len(dataset)
    avg_dict_count["all"]["nltk_word_avg"] = params[0]/len(dataset)
    avg_dict_count["all"]["nltk_sentence_avg"] = params[1]/len(dataset)
    avg_dict_count["all"]["punctutation_avg"] = params[2]/len(dataset)
    avg_dict_count["all"]["ascii_letters_avg"] = params[3]/len(dataset)
    avg_dict_count["all"]["ascii_lowercase_avg"] = params[4]/len(dataset)
    avg_dict_count["all"]["ascii_uppercase_avg"] = params[5]/len(dataset)
    avg_dict_count["all"]["digits_avg"] = params[6]/len(dataset)
    avg_dict_count["all"]["whitespace_avg"] = params[7]/len(dataset)
    avg_dict_count["all"]["word_avg"] = params[8]/len(dataset)
    avg_dict_count["all"]["line_avg"] = params[9]/len(dataset)

    json.dump(dict_count, count_dictionaries, ensure_ascii=False)
    count_dictionaries.write("\n")

    total_json_object = json.dumps(avg_dict_count, indent=2)
    with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-Oscar/summary_oscar_data_formatted.json", "w") as outfile:
        outfile.write(total_json_object)

    formatted_dataset.close()
    count_dictionaries.close()

if __name__ == "__main__":
    main()
