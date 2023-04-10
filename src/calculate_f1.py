import json
import os

from evaluation.evaluator import *

def calculate_f1(output_dir, tasks=('RE','EE','NER')):
    EvaluatorDict = {
        'RE':EvaluatorRE,
        'EE':EvaluatorEvent,
        'NER':EvaluatorNER
    }
    task_dict = {task: dict() for task in tasks}
    task_path = os.path.join(output_dir, 'predict_eval_predictions.jsonl')
    report_dir = os.path.join(output_dir, 'report')
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

    # export tsv
    for task_name, eval_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        scores = []
        for dataset_name, evaluator in eval_dict.items():
            evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
            rows.append((dataset_name, evaluator.get_metric()))
            scores.append(evaluator.get_metric())
        rows = sorted(rows, key=lambda x: x[0].lower())
        if len(scores) == 0:
            continue
        rows.append(('Average', sum(scores)/len(scores)))
        with open(os.path.join(report_dir, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f'{row[0]}\t{row[1]}\n')
                print('%48s\t%g'%row)

if __name__ == '__main__':
    root = '../output/davinc'
    os.environ['RANDOM_RECORD'] = '1'   # 是否开启随机记录
    os.environ['EXPORT_IMG'] = '0'      # 是否导出混淆矩阵图片
    calculate_f1(root)