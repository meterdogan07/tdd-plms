#!/bin/sh
#SBATCH --job-name=bartTR-encode_1024
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --output=process-data_%J.out
#SBATCH --mail-user=eacikgoz17@ku.edu.tr
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load anaconda/3.6
source activate eacikgoz17

echo 'number of processors:'$(nproc)

LENGTH=1024
TOKLENGTH=4096
f=/kuacc/users/eacikgoz17/tdd/tdd-corpus-00-1024.jsonl

echo $f
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json /kuacc/users/eacikgoz17/tdd/tdd-plms/tokenization/bart/Tokenizers/vocab.json \
    --vocab-bpe /kuacc/users/eacikgoz17/tdd/tdd-plms/tokenization/bart/Tokenizers/merges.txt \
    --input $f \
    --output pile-tr-bpe/$(basename $f).bpe \
    --chunksize 64 \
    --workers 60 \
    --max_len $TOKLENGTH
