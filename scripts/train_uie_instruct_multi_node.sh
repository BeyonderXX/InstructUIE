#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

export MODEL_PATH=/mnt/data/user/zhou_weikang/model_cache/t5-base
export DATA_DIR=/mnt/data/user/xia_han/dataset/IE_data/NER_processed/
export TASK_DIR=/mnt/data/user/xia_han/dataset/IE_data/     # 无用参数
export HOSTFILE=/root/LLM/hostfile

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

# port=$(shuf -i25000-30000 -n1)
port=51419

# 3090 * 8 on t5-700M
deepspeed \
   --hostfile=$HOSTFILE \
   --master_port $port \
   --master_addr 10.176.50.36 \
   --no_ssh_check \
   --force_multi \
   --launcher pdsh \
   src/run_s2s_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_task_definition True \
   --num_pos_examples 2 \
   --num_neg_examples 0 \
   --add_explanation False \
   --tk_instruct False \
   --data_dir $DATA_DIR \
   --task_dir $TASK_DIR \
   --output_dir output/ \
   --overwrite_output_dir \
   --cache_dir ./cache/ \
   --overwrite_cache \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 2 \
   --learning_rate 5e-05 \
   --num_train_epochs 10 \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 500 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 3856 \
   --run_name t5-experiment \
   --deepspeed ds_configs/stage2.config \

# # a100*8 on t5-3b
# deepspeed --master_port $port src/run_s2s_uie.py \
#     --do_train \
#     --do_predict \
#     --predict_with_generate \
#     --model_name_or_path /root/MODELS/t5-3b \
#     --max_source_length 512 \
#     --max_target_length 128 \
#     --generation_max_length 128 \
#     --max_num_instances_per_task 20000 \
#     --max_num_instances_per_eval_task 200 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 2 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir /root/InstructUIE/IE_data/NER_processed/ \
#     --task_dir /root/InstructUIE/data/tasks/ \
#     --output_dir output/ \
#     --overwrite_output_dir \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 5 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy steps \
#     --save_steps 2000 \
#     --deepspeed configs/stage2.config \
#     --bf16 \
#     --run_name t5-experiment
