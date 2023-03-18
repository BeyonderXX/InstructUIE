#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)


# 3090 * 8 on t5-700M
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port $port src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /root/MODELS/codegen-350M-mono \
   --data_dir /workspace/IE_data_v2 \
   --task_config_dir /workspace/InstructUIE/configs/multi_task_configs \
   --instruction_file /workspace/InstructUIE/configs/instruction_config.json \
   --output_dir output/codegen-350M-ie-test \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 6 \
   --learning_rate 5e-05 \
   --num_train_epochs 5 \
   --input_record_file codegen.record \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name codegen-350M-ie-single-experiment \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 100 \
   --evaluation_strategy steps \
   --evaluation_steps 2000 \
   --save_strategy steps \
   --save_steps 2000
