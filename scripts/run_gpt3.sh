export openai_key="Your OpenAI API key"

data_dir=data/splits/default
task_dir=data/tasks
output_dir=output/
max_num_instances_per_eval_task=20

echo "instruction + 2 positive examples"
for engine in "text-davinci-001" "davinci" 
do
echo $engine
python src/run_gpt3.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/default/gpt3/${engine}
python src/compute_metrics.py --predictions ${output_dir}/default/gpt3/${engine}/predicted_examples.jsonl --track default
done

echo "xlingual instruction + 2 positive examples"
for engine in "text-davinci-001" "davinci"
do
echo $engine
python src/run_gpt3.py \
    --data_dir data/splits/xlingual/ \
    --task_dir $task_dir \
    --overwrite_cache \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task ${max_num_instances_per_eval_task} \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --engine ${engine} \
    --output_dir ${output_dir}/xlingual/gp3/${engine}
python src/compute_metrics.py --predictions ${output_dir}/xlingual/gpt3/${engine}/predicted_examples.jsonl --track xlingual
done
