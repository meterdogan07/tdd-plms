import pandas as pd
import numpy as np
import csv
import os

dir = "/kuacc/users/abond19/COMP442/data_tooling/ac_dc/deduplicate/outputs/tr"

matches = pd.read_csv(os.path.join(dir, "matches.tsv"), sep='\t')
clusters = pd.read_csv(os.path.join(dir, "clusters.tsv"), sep='\t')

#print(clusters.iloc[0, :])

i = 0

output_file = open("deduped_text.csv", "w")

cluster_dict = dict()

with open("text.csv") as text:
    for line in text.readlines():
        element = clusters.iloc[i, :]['cluster']
        if element != -1:
            if element not in cluster_dict:
                new_list = [line]
                cluster_dict[element] = new_list
            else:
                cluster_dict[element].append(line)
        else:
            output_file.write(line)
        i += 1
        
        if i % 10000 == 0:
            print(f"Current line: {i}\n")
    
for key in cluster_dict.keys():
    output_file.write(cluster_dict[key][0])

output_file.close()