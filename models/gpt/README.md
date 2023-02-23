## Megatron-DeepSpeed
DeepSpeed version of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others.


Below is Megatron-LM's original README:
------

Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. They developed efficient, model-parallel (tensor and pipeline), and multi-node pre-training of transformer based models such as [GPT](https://arxiv.org/abs/2005.14165), [BERT](https://arxiv.org/pdf/1810.04805.pdf), and [T5](https://arxiv.org/abs/1910.10683) using mixed precision.

This codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism. To demonstrate how the code scales with multiple GPUs and model sizes, they consider GPT models from 1 billion all the way to 1 trillion parameters. All models use a vocabulary size of 51,200 and a sequence length of 2048. We vary hidden size, number of attention heads, and number of layers to arrive at a specifc model size. As the model size increases, we also modestly increase the batch size. We leverage [NVIDIA's Selene supercomputer](https://www.top500.org/system/179842/) to perform scaling studies and use up to 3072 [A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for the largest model. 

All the cases from 1 billion to 1 trillion parameters achieve more than 43% half precision utilization, which is high for an end-to-end application. We observe that initially the utilization remains constant but as hidden size increases for larger models, utilization starts increasing and reaches 52% for the largest model. We also note that achieved aggregate petaFLOPs across all GPUs increases almost linearly with number of GPUs, demonstrating good weak scaling.


# Contents
   * [Contents](#contents)
   * [Setup](#setup)
      * [Downloading Checkpoints](#downloading-checkpoints)
   * [Usage](#usage)
   * [Training](#training)
      * [Data Preprocessing](#data-preprocessing)
      * [BERT Pretraining](#bert-pretraining)
      * [GPT Pretraining](#gpt-pretraining)
      * [GPT Pretraining](#gpt-pretraining)
      * [T5 Pretraining](#t5-pretraining)
      * [Distributed Pretraining](#distributed-pretraining)
      * [GPT-3 Example](#gpt-3-example)
   * [Evaluation and Tasks](#evaluation-and-tasks)
      * [GPT Text Generation](#gpt-text-generation)
      * [GPT Evaluation](#gpt-evaluation)
         * [WikiText Perplexity Evaluation](#wikitext-perplexity-evaluation)
         * [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
      * [BERT Task Evaluation](#bert-task-evaluation)
         * [RACE Evaluation](#race-evaluation)
         * [MNLI Evaluation](#mnli-evaluation)
   * [Datasets](#datasets)
      * [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)
      * [Collecting GPT Webtext Data](#collecting-gpt-webtext-data)

# Setup
Install the latest supported versions of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). It is strongly recommend using one of [NGC's recent PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) (the latest compatible version at time of publication can be pulled with `docker pull nvcr.io/nvidia/pytorch:20.12-py3`). Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

The models require vocabulary files to run. The BERT  WordPiece vocab file can be extracted from Google's pretrained BERT models: [uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

Additional notes for DeepSpeed. We have added a helper script to download the checkpoints and make the example runnable.

Steps to follow:
 - bash dataset/download_ckpt.sh -- this will download and extract the checkpoint
 - bash dataset/download_vocab.sh -- this will download GPT merges and vocab files.
 - bash examples/generate_text.sh -- this will generate examples using the 345m GPT model.

# Usage

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

There are provided several scripts for pretraining both BERT and GPT in [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

# Training
## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for GPT training. Some minor modifications are required for GPT data preprocessing compared to BERT, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).


## GPT Pretraining

The `examples/pretrain_gpt.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

<pre>
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT_ARGS="--num-layers 24 \
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
          --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
</pre>

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).




## Distributed Pretraining

The `examples/pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables and using `init_method='env://'` in the launcher. See the official PyTorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the Python flag `-m torch.distributed.launch`, detailed below, are the only additional requirements to adopt distributed training.

They use two types of parallelism: data and model parallelism. We facilitate two distributed data parallel implementations: a simple one of our own that performs gradient all-reduce at the end of back propagation step, and Torch's distributed data parallel wrapper that overlaps gradient reduction with back propagation computation. To switch between these two options use `--DDP-impl local` or `--DDP-impl torch`, respectively. As expected, Torch distributed data parallelism is more efficient at larger model sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 76% when Torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

Second, they developed a simple and efficient two-dimensional model-parallel approach. To use tensor model parallelism (splitting execution of a single transformer module over multiple GPUs), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

There are examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`.
Other than these minor changes, the distributed training is identical to the training on a single GPU.

Distributed training:
<pre>
WORLD_SIZE=8
TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=&#60;same as above&#62;
VOCAB_FILE=&#60;same as above&#62;
DATA_PATH=&#60;same as above&#62;
MODEL_ARGS=&#60;same as above&#62;
OUTPUT_ARGS=&#60;same as above&#62;

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_<model>.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $TENSOR_MP_SIZE \
                --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
                --DDP-impl torch
</pre>

## GPT-3 Example

In `examples/pretrain_gpt3_175B.sh` we have provided an example of how to configure Megatron to run [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way and 16-way tensor and pipeline parallelism, respectively. With options `global-batch-size 1536` and `rampup-batch-size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incrmeental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.



# Evaluation and Tasks

They provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on a single GPU in downstream tasks. The following script accomplishes this. Currently only tensor model parallelism is supported on input and pipeline model parallelsim on the output. This example reads in a model with 2-way tensor model parallelism and writes out a model with 2-way pipeline model parallelism.

<pre>
TENSOR_MODEL_PARALLEL_SIZE=2
TARGET_PIPELINE_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE python tools/merge_mp_partitions.py \
        --model-type BERT \
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
        --pipeline-model-parallel-size 1 \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --tokenizer-type BertWordPieceLowerCase \
        --vocab-file $VOCAB_FILE \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH
        --save $CHECKPOINT_PATH/merged

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Text Generation
`bash examples/generate_text.sh`

They generate text samples using largely the GPT pretraining script. Few changes need to make, such as we need to provide the path to the pretrained checkpoint, the length of the output samples, whether to generate texts unconditionally (`--num-samples` to denote how many samples to generate) or conditional (need to pass `--sample-input-file <filename>` where each line of the file will be used as the conditional texts). There are few optional parameters to play, e.g. `top-k`, `top-p`, or `greedy` (set top-k and top-p to 0) sampling..

<pre>
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
GPT_ARGS=&#60;same as those in <a href="#gpt-pretraining">GPT pretraining</a> above&#62;

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
       $GPT_ARGS \
       --load $CHECKPOINT_PATH \
       --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
       --temperature $TEMPERATURE \
       --genfile $OUTPUT_FILE \
       --num-samples $NUMBER_OF_SAMPLES \
       --top_p $TOP_P \
       --recompute
</pre>

## GPT Evaluation
They include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

Use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --checkpoint-activations \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

# Datasets
We do not host any datasets for GPT or BERT training, however, Here is  detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, use the `--split-sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.

## Collecting GPT Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.
