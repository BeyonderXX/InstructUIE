#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# 注意 保证模型保存文件夹路径中包含模型文件名
# 注意 resume_from_checkpoint 和 model_name_or_path 保持一致

###########
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_s2s_uie_multitask_decoder.py \
    --do_predict \
    --resume_from_checkpoint $1 \
    --predict_with_generate \
    --model_name_or_path $1 \
    --max_source_length 768 \
    --max_target_length 768 \
    --generation_max_length 768 \
    --max_num_instances_per_task 50 \
    --max_num_instances_per_eval_task -1\
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir ./IE_data_v1 \
    --task_dir $3 \
    --output_dir $2 \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 32 \
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
    --deepspeed ds_configs/eval.config \
    --bf16 \
    --run_name flan-t5-700M-experiment

# For debug
#python src/run_s2s_uie_multitask_decoder.py \
#    --do_predict \
#    --resume_from_checkpoint $1 \
#    --predict_with_generate \
#    --model_name_or_path $1 \
#    --max_source_length 768 \
#    --max_target_length 768 \
#    --generation_max_length 768 \
#    --max_num_instances_per_task 50 \
#    --max_num_instances_per_eval_task -1 \
#    --add_task_name False \
#    --add_task_definition True \
#    --num_pos_examples 0 \
#    --num_neg_examples 0 \
#    --add_explanation False \
#    --tk_instruct False \
#    --data_dir ./IE_data_v1 \
#    --task_dir $3 \
#    --output_dir $2 \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --per_device_train_batch_size 20 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 2 \
#    --learning_rate 5e-05 \
#    --num_train_epochs 5 \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 500 \
#    --evaluation_strategy no \
#    --save_strategy steps \
#    --save_steps 2000 \
#    --bf16 \
#    --run_name flan-t5-700M-experiment

#########
# CUDA_VISIBLE_DEVICES=1,2,3,6 deepspeed --master_port $port src/run_s2s_uie_multitask.py \
#     --do_predict \
#     --resume_from_checkpoint ./output/flan-t5-700M-multi/checkpoint-42000 \
#     --predict_with_generate \
#     --model_name_or_path /mnt/data/user/zhou_weikang/model_cache/flan-t5-large \
#     --max_source_length 512 \
#     --max_target_length 128 \
#     --generation_max_length 50 \
#     --max_num_instances_per_task 50 \
#     --max_num_instances_per_eval_task -1\
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir ./IE_data \
#     --task_dir EE \
#     --output_dir ./output/flan-t5-700M-multi/output_42000/EE \
#     --overwrite_output_dir \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
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
#     --deepspeed configs/eval.config \
#     --run_name flan-t5-experiment


# a100*8 on t5-3b
# deepspeed --master_port $port src/run_s2s_uie_eval.py \
#     --do_predict \
#     --predict_with_generate \
#     --output_dir $1 \
#     --resume_from_checkpoint $2 \
#     --model_name_or_path $3 \
#     --max_source_length 512 \
#     --max_target_length 50 \
#     --generation_max_length 50 \
#     --max_num_instances_per_task 20 \
#     --max_num_instances_per_eval_task -1 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir ./IE_data/NER_processed/ \
#     --task_dir ./data/tasks/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 90 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 5 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy steps \
#     --save_steps 2000 \
#     --deepspeed ./configs/eval.config \
#     --run_name flan-t5-700M-experiment


# # a100*8 on t5-3b
# deepspeed --master_port $port src/run_s2s_uie_eval.py \
#     --do_predict \
#     --predict_with_generate \
#     --output_dir ./output/flan-t5-700M/output_30000 \
#     --resume_from_checkpoint ./output/flan-t5-700M/checkpoint-30000 \
#     --model_name_or_path /root/MODELS/flan-t5-700M \
#     --max_source_length 512 \
#     --max_target_length 50 \
#     --generation_max_length 50 \
#     --max_num_instances_per_task 20 \
#     --max_num_instances_per_eval_task -1 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 0 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir ./IE_data/NER_processed/ \
#     --task_dir ./data/tasks/ \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 90 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 5 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 500 \
#     --evaluation_strategy no \
#     --save_strategy steps \
#     --save_steps 2000 \
#     --deepspeed ./configs/eval.config \
#     --run_name flan-t5-700M-experiment
