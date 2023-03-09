import json

from evaluation.eval_main import eval

# task_path = r'/workspace/InstructUIE/output/flan-t5-700M/output_8000/RE/predicted_examples.jsonl'
# with open(task_path,'r') as f:
#     data_lines = [json.loads(line) for line in f.readlines()]
# json_list , pred_list = [], []
# for instance in data_lines:
#     json_list.append(instance)
#     pred_list.append(instance['prediction'])
# result = eval(json_list, pred_list, 'RE')
# print(result)

#######multi########
task_path = r'../output/codegen-350M-output/output_24000/RE/predicted_examples.jsonl'
with open(task_path,'r') as f:
    data_lines = [json.loads(line) for line in f.readlines()]
json_list , pred_list = [], []
groups = {}
for instance in data_lines:
    category = instance.get("Task")
    if category not in groups:
        groups[category] = []
    groups[category].append(instance)

# groups = dict(sorted(groups.items()))
for task in groups:   
    print(task)
for task in groups:   
    for instance in groups[task]:
        json_list.append(instance)
        pred_list.append(instance['prediction'])
    result = eval(json_list, pred_list, 'RE')
    print(result)