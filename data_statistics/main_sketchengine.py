from multiprocessing import cpu_count
from datasets import load_dataset
import os
import argparse
import pandas as pd
import json
import string
import codecs
import nltk
import nltk.tokenize
import numpy as np

def main():
    nltk.download('punkt')
    dir_name = "/userfiles/merdogan18/trTenTen12.vert"
    formatted_dataset = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-SketchEngine/sketchengine_data_formatted.json", "a")
    count_dictionaries = open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-SketchEngine/sketchengine_stats_formatted.json", "a")
    
    f = codecs.open(dir_name, 'r', encoding='utf-8', errors='ignore')
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    ct_doc = 0
    ct_line = 0
    next = False
    dict = {}
    dict_count = {}
    avg_dict_count = {}
    params = [0,0,0,0,0,0,0,0,0,0,0]
    max_sample = 11871272
    params_array = np.zeros((11,max_sample))
    
    txt = ""
    for line in f:
        ct_line+=1
        line_strip = line.strip()
        if(line_strip == "<g/>"): 
            next = True
            continue
        if(line_strip == "</p>"): 
            #print("p tag p")
            continue
        if(line_strip[0:10] == "<p heading"): 
            continue
        if(line_strip[0:10] == "<gap words"): 
            continue
        if(line[0:4] == "<doc"):
            dict["source"] = line_strip[line_strip.find("url"):]
            dict["doc_id"] = ct_doc
            dict["language"] = "tr"
            dict["corpus"] = "SketchEngine"
            dict["context"] = "null"
            ct_doc+=1
            print(ct_doc)
            continue
        if(line_strip == "</doc>"):
            txt = txt[1:]
            dict["text"] = txt
            i = ct_doc-1
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

            json.dump(dict,formatted_dataset, ensure_ascii=False)
            formatted_dataset.write("\n")
            
            dict = {}
            txt = ""
            continue
        if(line_strip in "{[(“"):
            next = True
            txt += " "+line_strip
            continue
        if(line_strip in ";:.,!?!\"]}”)" or next):
            txt += line_strip
            next = False
            continue
        txt += " "+line_strip
        
        if(ct_doc > max_sample):
            break
    f.close

    ct_doc = ct_doc-1
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
    avg_dict_count["all"]["size(GB)"] = os.path.getsize(dir_name)/(1024*1024*1024)
    avg_dict_count["all"]["total_samples"] = ct_doc

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

    avg_dict_count["all"]["length_avg"] = params[0]/ct_doc
    avg_dict_count["all"]["nltk_word_avg"] = params[0]/ct_doc
    avg_dict_count["all"]["nltk_sentence_avg"] = params[1]/ct_doc
    avg_dict_count["all"]["punctutation_avg"] = params[2]/ct_doc
    avg_dict_count["all"]["ascii_letters_avg"] = params[3]/ct_doc
    avg_dict_count["all"]["ascii_lowercase_avg"] = params[4]/ct_doc
    avg_dict_count["all"]["ascii_uppercase_avg"] = params[5]/ct_doc
    avg_dict_count["all"]["digits_avg"] = params[6]/ct_doc
    avg_dict_count["all"]["whitespace_avg"] = params[7]/ct_doc
    avg_dict_count["all"]["word_avg"] = params[8]/ct_doc
    avg_dict_count["all"]["line_avg"] = params[9]/ct_doc

    json.dump(dict_count, count_dictionaries, ensure_ascii=False)
    count_dictionaries.write("\n")

    total_json_object = json.dumps(avg_dict_count, indent=2)
    with open("/kuacc/users/merdogan18/hpc_run/tdd-plms/data_statistics/Processed-SketchEngine/summary_sketchengine_stats_formatted.json", "w") as outfile:
        outfile.write(total_json_object)
    
if __name__ == "__main__":
    main()
                         