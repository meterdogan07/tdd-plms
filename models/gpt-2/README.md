# Workspace GPT-2
Pre-training [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) for Turkish.

## GPT-2: Language Models are Unsupervised Multitask Learners
We want to pre-train an auto-regressive GPT-2 language model with a decoder-only architecture for Turkish language. We will follow [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) repository and its pipeline scheme to pretrain our model, which is a fork of original [Megatron-LM](https://github.com/bigscience-workshop/Megatron-DeepSpeed) repository. See [here](https://arxiv.org/pdf/1909.08053.pdf) for details of Megatron-LM.

### Usage

The most comprehensive flow is:
1. Installation and Setting Environment
2. Data Preprocessing
3. Pretraining
4. Finetunning (Optional for zero-shot tasks)
5. Downstream Task Evaluation or Text Generation

Several scripts are provided for pretraining GPT in examples directory.

## 1. Installation and Setting Environment

1. Install `bigscience-workshop/Megatron-DeepSpeed`
```
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed
cd Megatron-DeepSpeed
pip install -r requirements.txt
```

2. Install `apex`

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```

3. Install `deepspeed`

```
git clone https://github.com/microsoft/deepspeed
cd deepspeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
```

adjust `TORCH_CUDA_ARCH_LIST="7.0"` to the architecture of your NVIDIA GPU (or just remove it altogether if you are not sure how to find one).

3. CUDA kernels compilation

The first time you run the training scripts several CUDA kernels will be compiled. Which means you need to have a cuda environment set up in your environment and it should match the version pytorch was built with.



## 2. Data Preprocessing

**Vocab (Optional):** The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

1. The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
```
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training

2. The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`).

3. An example script to prepare data for GPT training is:

```
python tools/preprocess_data.py \
    --input my-corpus.json \
    --output-prefix my-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

The output will be two files named, in this case, `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. The `--data-path` specified in later GPT training is the full path and new filename, but without the file extension.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

You can also use `tools/preprocess_data_many_cores.py` in the case of high amount of cpu cores available. Typically in JZ setup where cpu nodes have up to 40 physical cpu cores, you should run this script with around 60 workers instead of the `tools/preprocess_data.py`. The same command line arguments are available.

4. Sometimes it's hard to work on a very large dataset at once, so one can pre-process it in chunks and then merge those datasets into a single combined indexed dataset. Here is an example:

```
python tools/merge_preprocessed_data.py \
    --datasets \
    meg-gpt2-oscar-en-500-p1_text_document \
    meg-gpt2-oscar-en-500-p2_text_document \
    meg-gpt2-oscar-en-500-p3_text_document \
    --output-prefix meg-gpt2_oscar_text_document
```

### Quick pre-processing to start training with

Here is how you can get ready to train quickly, using a 1GB 79K-record jsonl dataset.

```
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

## 3. Pretraining
The `examples/pretrain_gpt.sh` script runs single GPU 345M parameter GPT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--lr-warmup-fraction`. While this is single GPU training, the batch size specified by `--micro-batch-size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global-batch-size` whcih is the batch size per iteration.

The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train-iters` as the training iterations requested. Alternatively, one can provide `--train-samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr-decay-iters`, one will need to provide `--lr-decay-samples`.

The logging, checkpoint-saving, and evaluation intervals are specified. Checkpointing the activations facilitates the training of larger models and/or batches. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

The tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file), the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

However, as you will see below you will learn that DeepSpeed requires a distributed enviroment even with a single GPU. Therefore, **instead refer to [pretrain_gpt_single_node.sh](example/pretrain_gpt_single_node.sh), which will work with this repo**.

```
CHECKPOINT_PATH=checkpoints/gpt2
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --lr-warmup-fraction .01 \
    --fp16 \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    "

CMD="pretrain_gpt.py $GPT_ARGS $OUTPUT_ARGS $DATA_ARGS"

N_GPUS=1

LAUNCHER="deepspeed --num_gpus $N_GPUS"

$LAUNCHER $CMD
```

Note, we replaced `python` with `deepspeed --num_gpus 1`. For multi-gpu training update `--num_gpus` to the number of GPUs you have.

For multi-node training you will either need to create a `hostfile` which defines all the nodes as explained [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) or in the SLURM environment it might not work and you will need to use:

```
CMD=<as above>

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print'`
MASTER_PORT=6000
GPUS_PER_NODE=4
NNODES=16

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
```

For a single GPU the other approach is to emulate `distributed` with:
```
MASTER_ADDR=localhost MASTER_PORT=9994 RANK=0 LOCAL_RANK=0 python pretrain_gpt.py ...
```
Further command line arguments are described in the source file [`arguments.py`](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/arguments.py).

### Using any pretrained tokenizer

Thanks to @sbmaruf, any HF pretrained tokenizer may be used instead of the Megatron-provided BERT/GPT/T5 tokenizers. You'll need to run preprocessing yourself (`tools/preprocess_data.py`), using `tokenizer-type=PretrainedFromHF` and `tokenizer-name-or-path=<your_tokenizer>`. For example, `python tools/preprocess_data.py --input ~/c4_en_train.jsonl --output-prefix c4_en_train --dataset-impl mmap --tokenizer-type PretrainedFromHF --tokenizer-name-or-path t5-small --workers 30 --append-eod`



## 4. Finetunning (Optional for zero-shot tasks)
TO DO

## 5. Downstream Task Evaluation or Text Generation
TO DO
