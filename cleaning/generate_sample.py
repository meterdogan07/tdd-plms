from tqdm import tqdm

file = open("final/combined.txt")

i = 0

output_file = open("samples/combined_sample.txt", "w")

for element in tqdm(file):
    if i % 100000 == 0:
        output_file.write(element)
    i += 1

output_file.close()
        