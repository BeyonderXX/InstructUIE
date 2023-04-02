#!/bin/bash

CHECKPOINT_PATH=bloom-1b1-megatron_model

python3 tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1536 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 2048 \
       --fp16 \
       --micro-batch-size 2 \
       --seq-length 2048 \
       --out-seq-length 20 \
       --temperature 1.0 \
       --sample-input-file test_input.json \
       --sample-output-file test_output.json \
       --num-samples 0 \
       --pad-vocab-size-to 250880 \
       --embed-layernorm \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path bigscience/bloom \
       --top_p 0.9 \
       --recompute \
       --position-embedding-type alibi
