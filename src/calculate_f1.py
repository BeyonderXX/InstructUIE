import json
import os

from evaluation.evaluator import *

def calculate_f1(output_dir):
    EvaluatorDict = {
        'RE':EvaluatorRE,
        'EE':EvaluatorEvent,
        'NER':EvaluatorNER,
        'EET':EvaluatorEET,
        'EEA':EvaluatorEEA
    }
    task_dict = dict()      # str -> dict
    task_path = os.path.join(output_dir, 'predict_eval_predictions.jsonl')
    report_dir_root = os.path.join(output_dir, 'report')
    with open(task_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task_name = data['Task']
            dataset_name = data['Dataset']
            if task_name not in task_dict:
                task_dict[task_name] = dict()
            if dataset_name not in task_dict[task_name]:
                task_dict[task_name][dataset_name] = EvaluatorDict[task_name]()
            task_dict[task_name][dataset_name].add(data, data['Prediction'])

    # export report
    if not os.path.exists(report_dir_root):
        os.mkdir(report_dir_root)

    # export tsv
    for task_name, eval_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        scores = []
        report_dir = os.path.join(report_dir_root, task_name)
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        for dataset_name, evaluator in eval_dict.items():
            evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
            rows.append((dataset_name, evaluator.get_metric()))
            scores.append(evaluator.get_metric())
        rows = sorted(rows, key=lambda x: x[0].lower())
        if len(scores) == 0:
            continue
        rows.append(('Average', sum(scores)/len(scores)))
        with open(os.path.join(report_dir_root, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f'{row[0]}\t{row[1]}\n')
                print('%48s\t%g'%row)

if __name__ == '__main__':
    root = '../output/flant5-11b-v8-zeroshot'
    os.environ['RANDOM_RECORD'] = '1'   # 是否开启随机记录
    os.environ['EXPORT_IMG'] = '0'      # 是否导出混淆矩阵图片
    calculate_f1(root)