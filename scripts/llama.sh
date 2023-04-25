#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# 最好令 bs_per_gpu * num_gpu * gradient_accumulation_steps = 256
# 学习率可以使用 5e-5
# param_num < 1b 10epoch, 3b 5epoch, 11b 5epoch
# 注意修改 CUDA_VISIBLE_DEVICES, model_name_or_path，output_dir, run_name, data_dir, task_config_dir, instruction_file
# 其余参数可与当前版本保持一致

# 3090 * 4 on t5-700M
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /root/model_cache/llama-7b-hf \
   --data_dir /mnt/data/user/zhou_weikang/IE_data_v9_plus \
   --task_config_dir /workspace/InstructUIE/configs/debug \
   --instruction_file /workspace/InstructUIE-dev/configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/llama7b \
   --input_record_file llama.record \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 12 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-05 \
   --num_train_epochs 2 \
   --deepspeed configs/ds_configs/stage3.config \
   --run_name llama-mult-mi-experiment \
   --max_source_length 512 \
   --max_target_length 50 \
   --gradient_checkpointing \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name True \
   --add_dataset_name True \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 500 \
   --evaluation_strategy no \
   --save_strategy steps \
   --save_steps 10000
