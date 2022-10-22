# BART Tokenization

Pre-training a Turkish version of BART.

## Environment

- Install [Apex](https://github.com/NVIDIA/apex).
- Clone and install fairseq of Ali's version:

```
git clone https://github.com/alisafaya/fairseq.git
cd fairseq/
git checkout albart
export ROOT_DIR=$(pwd)
```

## Data

The main data location is: `/kuacc/users/asafaya19/tdd-corpora`. Use only `/kuacc/users/asafaya19/tdd-corpora/tdd-corpus-00.txt` for tokenization.

## Create 256/1024 versions
Use the code below to create the different context length files in your work dir:

```
mkdir tdd-corpora-1024-split/
mkdir tdd-corpora-256-split/
for file in /kuacc/users/asafaya19/tdd-corpora/*; do
    head -n 4500000 $file > tdd-corpora-256-split/$(basename $file)
    tail -n 450000 $file > tdd-corpora-1024-split/$(basename $file)
done
```

### Split Data

We split the `tdd-corpus-00.txt` in to 10 files to work with multiple CPUs quicker. Each tdd-corpus contains 500,000 lines. Use the following command to split files:
```
split --verbose -l 500000 -d --additional-suffix=".txt" /kuacc/users/asafaya19/tdd-corpora/tdd-corpus-00.txt output_name
```

## Training Tokenizer

We follow a similar procedure with RoBERTa pre-training procedure [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).

For training a Byte-level BPE tokenizer, just run build_bart_tokenizer.py script in `tokenization/bart/Tokenizers` directory. Please, do not forget to change the source data location with yours. It will result with a json object as `BPE.byte.tokenizer.your_vocab_size.json`. To get corresponding `vocab.json` and `merges.txt`, run `get_vocab_and_merges.py` by giving the your tokenizer's json file location.

## Encoding
In order to run apply sentence segmentation for 256 and and 1024 tokens, first you need to conver the related file to JSON lines format; for that please run `txt2jsonl.py`.

Then run `./encode_256.sh` and `./encode_1024.sh` by giving your own `BPE.byte.tokenizer.your_vocab_size.json`-`vocab.json`-`merges.txt` file locations to get the `fairseq/pile-tr-bpe/tdd-corpus-00-256.jsonl.bpe` and `fairseq/pile-tr-bpe/tdd-corpus-00-1024.jsonl.bpe` outputs.









