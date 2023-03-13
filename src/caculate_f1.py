import json

from evaluation.evaluator import *

task_path = r'../output/output_1200/RE/predicted_examples.jsonl'
Evaluator = EvaluatorRE

task_dict = dict()
with open(task_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        task_name = data['Task']
        if task_name not in task_dict:
            task_dict[task_name] = Evaluator()
        task_dict[task_name].add(data, data['prediction'])

for task_name, evaluator in task_dict.items():
    print(task_name, evaluator.get_metric(), sep='\t')
    evaluator.dump_audit_report(task_name+'.json')