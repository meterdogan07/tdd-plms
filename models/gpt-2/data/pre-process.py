import json, argparse, os
from tqdm import tqdm

parser = argparse.ArgumentParser("")
parser.add_argument("--file_path", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
args = parser.parse_args()
print(args)

filename    = os.path.splitext(args.file_path)[0].split("/")[-1]
current_dir = os.getcwd()
output_dir  = os.path.join(current_dir, "new_data")
isExist     = os.path.exists(output_dir)
if not isExist:
    os.mkdir(output_dir)
output_filename = output_dir + "/" + filename + ".jsonl"

data_list = []
with open(args.file_path, "r", encoding="UTF-8") as f:
    lines = f.readlines()
    print(f"Reading the data at: {filename}")
    for idx, line in tqdm(enumerate(lines)):
        data = {"src": "The Internet", 
                "text": line.rstrip(), 
                "type": "Tur",
                "id": str(idx),
                "title": filename
                }
        data_list.append(data)

with open(output_filename, 'w', encoding="UTF-8") as outfile:
    print(f"Writing the data to: {output_filename}")
    for sample in tqdm(data_list):
        json.dump(sample, outfile, ensure_ascii=False)
        outfile.write('\n')
