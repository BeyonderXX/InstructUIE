#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)


# 3090 * 8 on t5-700M
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port ../src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /root/MODELS/gpt-neox-chat-base-20b/models--togethercomputer--GPT-NeoXT-Chat-Base-20B/snapshots/gpt-neox-chat-base-20b \
   --data_dir ../workspace/IE_data_v2 \
   --task_config_dir ../configs/multi_task_configs \
   --instruction_file ../configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir ../output/gpt-neox-20b-ie-single \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 6 \
   --learning_rate 5e-05 \
   --num_train_epochs 10 \
   --deepspeed ../configs/ds_configs/stage2.config \
   --run_name gpt-neox-20b-mult-test-experiment \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --instruction_strategy multiple \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 100 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 2000
