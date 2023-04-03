output_dir=output/test_convert
data_dir=/workspace/IE_data_v3
task_config_dir=/workspace/InstructUIE/configs/debug_configs
instruction_file=/workspace/InstructUIE/configs/instruction_config.json

python src/uie_dist_prepro.py \
    --data_dir $data_dir \
    --task_config_dir $task_config_dir \
    --instruction_file  $instruction_file \
    --instruction_strategy single \
    --processed_out_dir $output_dir \
    --max_num_instances_per_task 10000 \
    --max_num_instances_per_eval_task 200 \
    --add_task_name False \
    --add_dataset_name False \
    --num_examples 0 \
    --over_sampling False
