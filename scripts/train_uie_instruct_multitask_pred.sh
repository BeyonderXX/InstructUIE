#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# 3090 * 8 on t5-700M
#deepspeed --master_port $port src/run_s2s_uie.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path /root/MODELS/t5-700M \
#    --max_source_length 512 \
#    --max_target_length 128 \
#    --generation_max_length 128 \
#    --max_num_instances_per_task 10000 \
#    --max_num_instances_per_eval_task 200 \
#    --add_task_name False \
#    --add_task_definition True \
#    --num_pos_examples 2 \
#    --num_neg_examples 0 \
#    --add_explanation False \
#    --tk_instruct False \
#    --data_dir /root/InstructUIE/IE_data/NER_processed/ \
#    --task_dir /root/InstructUIE/data/tasks/ \
#    --output_dir output/ \
#    --overwrite_output_dir \
#    --cache_dir ./cache/ \
#    --overwrite_cache \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 2 \
#    --learning_rate 5e-05 \
#    --num_train_epochs 10 \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 500 \
#    --evaluation_strategy no \
#    --save_strategy steps \
#    --save_steps 3856 \
#    --deepspeed ds_configs/stage2.config \
#    --bf16 \
#    --run_name t5-experiment

# a100*8 on t5-3b
CUDA_VISIBLE_DEVICES=1,2,3,6 deepspeed --master_port $port src/run_s2s_uie_multitask.py \
    --do_predict_onebyone \
    --resume_from_checkpoint /workspace/InstructUIE/output \
    --predict_with_generate \
    --model_name_or_path /mnt/data/user/zhou_weikang/model_cache/t5-small \
    --max_source_length 100 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_num_instances_per_task 50 \
    --max_num_instances_per_eval_task 20 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir /workspace/InstructUIE/IE_data \
    --task_dir RE,NER \
    --output_dir output/t5-small-test_one \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --deepspeed ds_configs/stage0.config \
    --run_name t5-experiment
