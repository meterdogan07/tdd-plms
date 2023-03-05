# Deduplication

### Initialize Environment
```bash
git init
git pull https://github.com/google-research/deduplicate-text-datasets
```
### Download Dependencies
Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Install GCC
```bash
sudo apt-get install gcc
```
Install Other Dependencies
```bash
pip3 install numpy scipy tensorflow tensorflow_datasets transformers sentencepiece
```
### Compile the Code
```bash
cargo build
```

Before starting the deduplication process, create ```cache``` and ```out``` directories.
```bash
mkdir [path/to/dataset]/cache
```
```bash
mkdir [path/to/dataset]/out
```
### Start Deduplication

First, create a suffix array of the data:
```bash
python3 scripts/make_suffix_array.py [path/to/dataset]
```

Then,  
```bash
cargo run self-similar --data-file data/[path/to/dataset] --length-threshold [threshold] --cache-dir [path/to/dataset/cache] --num-threads [number of threads]
```
This will create 2 files (1 dups, 1 sizes file in the cache directory)

Finally, run: 
```bash
cargo run collect --data-file [path/to/dataset] --cache-dir [path/to/dataset/cache] --length-threshold [threshold] > [path/to/dataset/out/byte_range]
```
This will create a "byte_range" file in the "out" file (created above), which contains the byte ranges of strings which occur more than once. 

### DEMO

After creating the ```cache``` and ```out``` directories, run the following lines. ( ```demo_example``` contains 4 paragraphs, 2 of which are identical)
```bash
python3 scripts/make_suffix_array.py demo_data/demo_example
cargo run self-similar --data-file demo_data/demo_example --length-threshold 100 --cache-dir demo_data/cache --num-threads 1
cargo run collect --data-file demo_data/demo_example --cache-dir demo_data/cache --length-threshold 100 > demo_data/out/demo_byte_range
```
This code will create a txt file under ```demo_data/out```. The contents will look like the following:
```bash
S 9012
Merging.
out
0 2352
6230 8582
```
This means that the contents in the ranges [0:2352] and [6230:8582] are the same. 

To verify this: 
```bash
$ python3
>>> data=open("demo_data/demo_example","rb").read()
>>> data[0:2352]
b'Jordan B Peterson and Andrew Tate are two individuals with different backgrounds, perspectives, and areas of expertise. While there may be some similarities between them, there are also significant differences.\n\nJordan B Peterson is a clinical psychologist, professor of psychology, and author. He is best known for his views on personal responsibility, self-improvement, and the importance of tradition and individualism. Peterson gained a large following after his book "12 Rules for Life: An Antidote to Chaos" was published in 2018. He also gained notoriety for his opposition to political correctness, his criticism of postmodernism and Marxism, and his defense of free speech.\n\nAndrew Tate, on the other hand, is a retired kickboxer, entrepreneur, and social media personality. He is known for his controversial views on various topics, including relationships, masculinity, and the importance of money. Tate gained a following on social media, particularly on Instagram and YouTube, where he shares his views and offers advice to his followers. He also runs several businesses, including a webcam studio and a cryptocurrency trading platform.\n\nIn terms of their similarities, both Jordan B Peterson and Andrew Tate are known for their unconventional and controversial views. They are both charismatic speakers who are not afraid to speak their minds and challenge conventional wisdom. They also have a significant following on social media and are seen as role models by many of their followers.\n\nHowever, there are also significant differences between the two. Jordan B Peterson is an academic and a clinical psychologist, while Andrew Tate is a retired kickboxer and entrepreneur. Peterson\'s views are grounded in research and science, while Tate\'s views are more based on personal experience and opinion. Peterson\'s focus is on personal responsibility and self-improvement, while Tate\'s focus is on making money and achieving success.\n\nIn conclusion, while Jordan B Peterson and Andrew Tate share some similarities, such as their charismatic speaking styles and unconventional views, there are also significant differences between them. Peterson is an academic and clinical psychologist focused on personal responsibility and self-improvement, while Tate is an entrepreneur and former kickboxer focused on making money and achieving success.\n'
>>> data[6230:8582]
b'Jordan B Peterson and Andrew Tate are two individuals with different backgrounds, perspectives, and areas of expertise. While there may be some similarities between them, there are also significant differences.\n\nJordan B Peterson is a clinical psychologist, professor of psychology, and author. He is best known for his views on personal responsibility, self-improvement, and the importance of tradition and individualism. Peterson gained a large following after his book "12 Rules for Life: An Antidote to Chaos" was published in 2018. He also gained notoriety for his opposition to political correctness, his criticism of postmodernism and Marxism, and his defense of free speech.\n\nAndrew Tate, on the other hand, is a retired kickboxer, entrepreneur, and social media personality. He is known for his controversial views on various topics, including relationships, masculinity, and the importance of money. Tate gained a following on social media, particularly on Instagram and YouTube, where he shares his views and offers advice to his followers. He also runs several businesses, including a webcam studio and a cryptocurrency trading platform.\n\nIn terms of their similarities, both Jordan B Peterson and Andrew Tate are known for their unconventional and controversial views. They are both charismatic speakers who are not afraid to speak their minds and challenge conventional wisdom. They also have a significant following on social media and are seen as role models by many of their followers.\n\nHowever, there are also significant differences between the two. Jordan B Peterson is an academic and a clinical psychologist, while Andrew Tate is a retired kickboxer and entrepreneur. Peterson\'s views are grounded in research and science, while Tate\'s views are more based on personal experience and opinion. Peterson\'s focus is on personal responsibility and self-improvement, while Tate\'s focus is on making money and achieving success.\n\nIn conclusion, while Jordan B Peterson and Andrew Tate share some similarities, such as their charismatic speaking styles and unconventional views, there are also significant differences between them. Peterson is an academic and clinical psychologist focused on personal responsibility and self-improvement, while Tate is an entrepreneur and former kickboxer focused on making money and achieving success.\n'
```





