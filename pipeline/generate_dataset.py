import os
import json
from datasets import load_dataset


CURRENT_DIR = os.path.dirname(__file__)

'''
Arguments
'''
data_dir = "/mnt/bd/zenithbloomvol/data/IE_data_v3"
task_dir = "NER,RE,EE"
prompt_dir = "/mnt/bd/zenithbloomvol/data/IE_data_v3/prompt.json"
cache_dir = "/mnt/bd/zenithllamavol/cache"
output_dir = "/mnt/bd/zenithbloomvol/data/IE_data_v3"
max_num_instances_per_task = 10000
max_num_instances_per_eval_task = 200
do_train = True
do_eval = True
do_predict = False
max_train_samples = None
max_eval_samples = None
max_predict_samples = None
max_seq_len = 1024
add_name = False

'''
loading datasets (not modified)
'''
raw_datasets = load_dataset(
    os.path.join(CURRENT_DIR, "uie_dataset_multitask.py"),
    data_dir=data_dir, 
    task_dir=task_dir,
    prompt_dir=prompt_dir,
    cache_dir=cache_dir,
    max_num_instances_per_task=max_num_instances_per_task,
    max_num_instances_per_eval_task=max_num_instances_per_eval_task
)
print(raw_datasets)

'''
dividing datasets (not modified)
'''
if do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))
if do_eval:
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(max_eval_samples))
if do_predict:
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(max_predict_samples)) # 取raw_datasets中的前n个样本

def process_data(batch):    
    instructions = []
    inputs = []
    tasks = []
    datasets = []
    for instance in batch:
        instructions.append(instance["instruction"])
        inputs.append(instance['Instance']['sentence'])
        tasks.append(instance["Categories"])
        datasets.append(instance["Task"])
    model_inputs = {"task": [], "dataset": [], "instructions": [], "inputs": [], "labels": []}
    model_inputs["task"] = tasks
    model_inputs["dataset"] = datasets
    model_inputs["instructions"] = instructions
    model_inputs["inputs"] = inputs
    # 修改key
    if "entities" in batch[0]["Instance"] and batch[0]["Instance"]["entities"]:
        jsons = [json.loads(ex["Instance"]["entities"].replace("'", '"').replace("#$%#", "'")) for ex in batch]
        labels = []
        for entities in jsons:
            if entities:
                kv_pairs = []
                relation_pairs = []
                event_pairs = []
                # TODO， 针对任务分别封装，仅返回label结果
                for entity in entities:
                    # 分别处理NER和RE
                    if 'type' in entity and 'trigger' in entity and 'arguments' in entity:
                        event_type = entity['type']
                        event_trigger = entity['trigger']
                        event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for argument in entity['arguments']]
                        event_pairs_ = [event_type,event_trigger,event_arguments]
                        event_pairs.append(event_pairs_)
                    elif 'type' in entity and 'name' in entity:
                        kv_pairs_ = [entity['type'], entity['name']]
                        kv_pairs.append(kv_pairs_)
                    elif 'head' in entity and 'type' in entity and 'tail' in entity:
                        relation_pairs_ = [entity['head']['name'],entity['type'],entity['tail']['name']]
                        relation_pairs.append(relation_pairs_)
                if len(event_pairs)>0:
                    label = ", ".join(["(type:{}, trigger:{}, arguments:".format(type, trigger)+", ".join(arguments)+")" for (type, trigger, arguments) in event_pairs])
                elif len(kv_pairs)>0:
                    label = ", ".join(["({}, {})".format(v, k) for (k, v) in kv_pairs])
                elif len(relation_pairs)>0:
                    label = ", ".join(["({}, {}, {})".format(h,r,t) for (h,r,t) in relation_pairs])
                labels.append(label)
            else:
                labels.append("[]")
        model_inputs["labels"] = labels
    return model_inputs

train_input = process_data(train_dataset)
dev_input = process_data(eval_dataset)

# '''
# TencentPretrain
# '''
# with open("UIE_text2text/train.tsv", "w", encoding="utf-8") as file1:
    # file1.writelines("text_a" + "\t" + "label" + "\n")
    # for i in range(len(train_input["inputs"])):
        # file1.writelines(train_input["inputs"][i] + "\t" + train_input["labels"][i] + "\n")
# with open("UIE_text2text/dev.tsv", "w", encoding="utf-8") as file2:
    # file2.writelines("text_a" + "\t" + "label" + "\n")
    # for j in range(len(dev_input["inputs"])):
        # file2.writelines(dev_input["inputs"][j] + "\t" + dev_input["labels"][j] + "\n")

'''
ChatLLaMA
'''
with open(os.path.join(output_dir, "UIE_train.json"), "w", encoding="utf-8") as file1:
    train_set = []
    for i in range(len(train_input["inputs"])):
        task = train_input["task"][i]
        dataset = train_input["dataset"][i]
        instruction = train_input["instructions"][i]
        input = train_input["inputs"][i]
        completion = train_input["labels"][i]
        if add_name == True:
            user_input = "Task: " + task + "\nDataset: " + dataset + "\n" + instruction + "Text: " + input + "\nAnswer: "
        else:
            user_input = instruction + "Text: " + input + "\nAnswer: "
        if (len(user_input) + len(completion)) > max_seq_len - 5:
            continue
        train_set.append({"Task": task, "Dataset": dataset, "user_input": user_input, "completion": completion})
    json.dump(train_set, file1, ensure_ascii=False)
    print("trainset: ", len(train_set))
with open(os.path.join(output_dir, "UIE_dev.json"), "w", encoding="utf-8") as file2:
    dev_set = []
    for j in range(len(dev_input["inputs"])):
        task = dev_input["task"][j]
        dataset = dev_input["dataset"][j]
        instruction = dev_input["instructions"][j]
        input = dev_input["inputs"][j]
        completion = dev_input["labels"][j]
        if add_name == True:
            user_input = "Task: " + task + "\nDataset: " + dataset + "\n" + instruction + "Text: " + input + "\nAnswer: "
        else:
            user_input = instruction + "Text: " + input + "\nAnswer: "
        if (len(user_input) + len(completion)) > max_seq_len - 5:
            continue
        dev_set.append({"Task": task, "Dataset": dataset, "user_input": instruction + "Text: " + input + "\nAnswer: ", "completion": completion})
    json.dump(dev_set, file2, ensure_ascii=False) 
    print("devset: ", len(dev_set))

'''
Alpaca
'''
# with open("UIE_text2text/UIE_train_alpaca_mini.json", "w", encoding="utf-8") as file1:
#     train_set = []
#     for i in range(len(train_input["inputs"])):
#         instruction = train_input["instructions"][i]
#         input = train_input["inputs"][i]
#         output = train_input["labels"][i]
#         if (len(input)+len(instruction)+len(output)) >= max_seq_len: # 丢掉超出长度的样本
#             continue
#         train_set.append({"instruction": instruction, "input": input, "output": output})
#     json.dump(train_set, file1, ensure_ascii=False)
#     print("trainset: ", len(train_set))
# with open("UIE_text2text/UIE_dev_alpaca_mini.json", "w", encoding="utf-8") as file2:
#     dev_set = []
#     for j in range(len(dev_input["inputs"])):
#         instruction = dev_input["instructions"][j]
#         input = dev_input["inputs"][j]
#         output = dev_input["labels"][j]
#         if (len(input)+len(instruction)+len(output)) >= max_seq_len: 
#             continue
#         dev_set.append({"instruction": instruction, "input": input, "output": output})
#     json.dump(dev_set, file2, ensure_ascii=False) 
#     print("devset: ", len(dev_set))
