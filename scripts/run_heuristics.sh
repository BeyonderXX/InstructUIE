output_dir="output/"

echo "Copy_demo for English track"
python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo --output_dir ${output_dir}/default/copy_demo
python src/compute_metrics.py --predictions ${output_dir}/default/copy_demo/predicted_examples.jsonl --track default

echo "Copy_input for English track"
python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/default --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input --output_dir ${output_dir}/default/copy_input
python src/compute_metrics.py --predictions ${output_dir}/default/copy_input/predicted_examples.jsonl --track default

echo "Copy_demo for x-lingual track"
python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/xlingual --max_num_instances_per_task 1 --max_num_instances_per_eval_task 100 --method copy_demo --output_dir ${output_dir}/xlingual/copy_demo
python src/compute_metrics.py --predictions ${output_dir}/xlingual/copy_demo/predicted_examples.jsonl --track xlingual

echo "Copy_input for x-lingual track"
python src/run_heuristics.py --task_dir data/tasks --data_dir data/splits/xlingual --max_num_instances_per_task 0 --max_num_instances_per_eval_task 100 --method copy_input --output_dir ${output_dir}/xlingual/copy_input
python src/compute_metrics.py --predictions ${output_dir}/xlingual/copy_input/predicted_examples.jsonl --track xlingual