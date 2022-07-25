import numpy as np
import os
import matplotlib.pyplot as plt

file_names = ["BPE_64k", "BPE_32k", "Unigram_64k", "Unigram_32k"]
#file_names = ["BPE_64k"]

plt.rcParams["figure.figsize"] = [100, 50]
plt.rcParams["figure.autolayout"] = True
#plt.ylim(0, 1)

for file in file_names:
    f = np.genfromtxt(f"Frequencies/{file}.txt", delimiter=",", dtype=np.int32)
    plt.bar(f[:, 0], f[:, 1])
    plt.savefig(f"Frequencies/{file}.png")  
    plt.clf()