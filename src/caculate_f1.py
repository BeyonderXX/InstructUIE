import json

from evaluation.evaluator import *

task_path = r'../output/bloomz-7b1/predict_eval_predictions.jsonl'
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

for task_name, eval_dict in task_dict.items():
    print('[[[[%s]]]]'%task_name)
    for dataset_name, evaluator in eval_dict.items():
        print(dataset_name, evaluator.get_metric(), sep='\t')
        evaluator.dump_audit_report('../output/bloomz-7b1/report/'+dataset_name+'.json')