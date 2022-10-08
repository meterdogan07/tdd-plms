#!/bin/bash

# Adapted to use deepspeed on a single node
#
# Multi-node will require either a `hostfile` or switching to `torch.distributed.launch`

# adjust to the number of GPUs to use
N_GPUS=1

CHECKPOINT_PATH=checkpoints/gpt2
DATA_PATH=~/Corpus/tdd-plms/models/gpt-2/test_gpt2_text_document.bin

GPT_ARGS=" \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr-decay-iters 320000 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --train-iters 5000 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path pretrained_tokenizer/ \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
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

#ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS"

#LAUNCHER="deepspeed --num_gpus $N_GPUS"

#CMD="$LAUNCHER ~/Corpus/Megatron-DeepSpeed/pretrain_gpt.py $ALL_ARGS"

#echo $CMD

#$CMD

CMD="pretrain_gpt.py $GPT_ARGS $OUTPUT_ARGS $DATA_ARGS"

N_GPUS=1

LAUNCHER="deepspeed --num_gpus $N_GPUS"

$LAUNCHER $CMD
