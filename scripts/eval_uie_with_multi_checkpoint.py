import os


os.system('''set -x''')
os.system('''export CUDA_DEVICE_ORDER="PCI_BUS_ID"''')
os.system('''export TRANSFORMERS_CACHE=/root/.cache/huggingface''')

file = [
        '/mnt/data/user/wangxiao/InstructUIE/Cache/output/checkpoint-11568',
        '/mnt/data/user/wangxiao/InstructUIE/Cache/output/checkpoint-3856',
        '/mnt/data/user/wangxiao/InstructUIE/Cache/output/checkpoint-7712',
        ]
for file_name in file:

    output_file = file_name.split('/')[-1]+'-output'
    cmd = '''CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 25001 src/run_s2s_uie_eval.py \
    --do_predict \
    --predict_with_generate \
    --output_dir {output_file} \
    --resume_from_checkpoint {file_name} \
    --model_name_or_path /mnt/data/user/zhou_weikang/model_cache/t5-base \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --max_num_instances_per_task 20 \
    --max_num_instances_per_eval_task -1 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir /workspace/InstructUIE/data/NER_processed/ \
    --task_dir /workspace/InstructUIE/data/tasks/ \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 90 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --deepspeed configs/eval.config \
    --run_name t5-experiment
        '''.format(file_name = file_name,output_file=output_file)
    print(cmd)
    print('!!!!!!!!!!!!!')
    os.system(cmd)







