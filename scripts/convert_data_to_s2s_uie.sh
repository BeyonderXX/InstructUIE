output_dir=data/uie
data_dir=/root/InstructUIE/IE_data/NER_processed/
task_dir=/root/InstructUIE/data/tasks/

# zero shot
python src/convert_data_to_s2s_uie.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 20000 \
    --max_num_instances_per_eval_task 200 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 512 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_only/
