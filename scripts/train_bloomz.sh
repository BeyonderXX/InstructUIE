#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)


# 3090 * 8 on t5-700M
CUDA_VISIBLE_DEVICES=5,6 deepspeed --master_port $port src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /root/MODELS/bloomz-560m \
   --data_dir /workspace/IE_data_v3 \
   --task_config_dir /workspace/InstructUIE/configs/debug_configs \
   --instruction_file /workspace/InstructUIE/configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/bloomz-560m-ie-single \
   --input_record_file bloomz.record \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 16 \
   --gradient_accumulation_steps 6 \
   --gradient_checkpointing True \
   --learning_rate 5e-05 \
   --num_train_epochs 3 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name bloomz-560m-mult-test-experiment \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 100 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 2000


