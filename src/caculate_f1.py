import json
import os

from evaluation.evaluator import *

output_dir = '../output/bloomz-7b1/'

task_path = os.path.join(output_dir, 'predict_eval_predictions.jsonl')
report_dir = os.path.join(output_dir, 'report')

EvaluatorDict = {
    'RE':EvaluatorRE,
    'EE':EvaluatorEvent,
    'NER':EvaluatorNER
}

task_dict = {key: dict() for key in EvaluatorDict}
with open(task_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        task_name = data['Task']
        dataset_name = data['Dataset']
        Evaluator = EvaluatorDict[task_name]
        if dataset_name not in task_dict[task_name]:
            task_dict[task_name][dataset_name] = Evaluator()
        task_dict[task_name][dataset_name].add(data, data['Prediction'])

# export report
if not os.path.exists(report_dir):
    os.mkdir(report_dir)

for task_name, eval_dict in task_dict.items():
    print('[[[[%s]]]]'%task_name)
    scores = []
    for dataset_name, evaluator in eval_dict.items():
        scores.append(evaluator.get_metric())
        print(dataset_name, evaluator.get_metric(), sep='\t')
        evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
    print('Average:', sum(scores)/len(scores))

# export tsv
for task_name, eval_dict in task_dict.items():
    rows = []
    scores = []
    for dataset_name, evaluator in eval_dict.items():
        rows.append((dataset_name, evaluator.get_metric()))
        scores.append(evaluator.get_metric())
    rows = sorted(rows, key=lambda x: x[0])
    rows.append(('Average', sum(scores)/len(scores)))
    with open(os.path.join(report_dir, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(f'{row[0]}\t{row[1]}\n')