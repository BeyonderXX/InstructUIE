import os
import json
import torch
import time
from typing import Optional, Tuple
from datasets import load_dataset


CURRENT_DIR = os.path.dirname(__file__)

'''
Arguments
'''
# 以下是load_dataset的参数
data_dir = "./IE_data_v3" # 数据集目录路径
task_config_dir = "./v3_configs" # 设置使用哪些数据集, 目录路径
instruction_dir = "./prompt.json" # prompt文件路径
instruction_strategy = "single" # single表示使用固定的指令, multiple表示随机选择指令
cache_dir = "./cache/" 
max_num_instances_per_task = 10000 # 每个数据集的最大样本数. 如果为None, 则使用全部
max_num_instances_per_eval_task = 100 # 同上
num_examples = 0 # 上下文学习的样例数量. [To Do]暂不支持few-shot
over_sampling = True # 是否过采样到max_num_instances

do_train = True # 是否输出训练集
do_eval = True # 是否输出验证集
do_predict = False # 是否输出测试集
max_train_samples = None # 在数据集中随机采样x条数据. 如果为None, 则使用全部
max_eval_samples = None # 同上
max_predict_samples = None # 同上
max_seq_len = 1024 # 最大生成长度. 不超过1024
trainset_dir = "UIE_dataset/UIE_train.json" # 输出的训练集路径
devset_dir = "UIE_dataset/UIE_dev.json" # 输出的验证集路径
testset_dir = "UIE_dataset/UIE_test.json" # 输出的测试集路径

'''
loading datasets (not modified)
belike:
    DatasetDict({
        train: Dataset({
            features: ['Task', 'Dataset', 'subset', 'Samples', 'Instance'],
            num_rows: 114
        })
        validation: Dataset({
            features: ['Task', 'Dataset', 'subset', 'Samples', 'Instance'],
            num_rows: 514
        })
        test: Dataset({
            features: ['Task', 'Dataset', 'subset', 'Samples', 'Instance'],
            num_rows: 1919
        })
    })
'''
raw_datasets = load_dataset(
    os.path.join(CURRENT_DIR, "uie_dataset.py"),
    data_dir=data_dir, 
    task_config_dir=task_config_dir,
    instruction_file=instruction_dir,
    instruction_strategy=instruction_strategy,
    cache_dir=cache_dir,
    max_num_instances_per_task=max_num_instances_per_task,
    max_num_instances_per_eval_task=max_num_instances_per_eval_task,
    num_examples=num_examples,
    over_sampling=over_sampling
)

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
    labels = []
    for instance in batch:
        instructions.append(instance["Instance"]["instruction"])
        inputs.append(instance["Instance"]["sentence"])
        tasks.append(instance["Task"])
        datasets.append(instance["Dataset"])
        labels.append(instance["Instance"]["ground_truth"])
    model_inputs = {"task": [], "dataset": [], "instructions": [], "inputs": [], "labels": []}
    model_inputs["task"] = tasks
    model_inputs["dataset"] = datasets
    model_inputs["instructions"] = instructions
    model_inputs["inputs"] = inputs
    model_inputs["labels"] = labels
    return model_inputs

if do_train:
    train_input = process_data(train_dataset)
if do_eval:
    dev_input = process_data(eval_dataset)
if do_predict:
    test_input = process_data(predict_dataset)

'''
rawData -> ChatLLaMA
'''
if do_train:
    with open(trainset_dir, "w", encoding="utf-8") as file1:
        train_set = []
        for i in range(len(train_input["inputs"])):
            task = train_input["task"][i]
            dataset = train_input["dataset"][i]
            instruction = train_input["instructions"][i]
            input = train_input["inputs"][i]
            completion = train_input["labels"][i]
            user_input = "Task: " + task + "\nDataset: ": dataset + "\n" + instruction + "Text: " + input + "\nAnswer: "
            if (len(user_input) + len(completion)) > max_seq_len - 5:
                continue
            train_set.append({"Task": task, "Dataset": dataset, "user_input": user_input, "completion": completion})
        json.dump(train_set, file1, ensure_ascii=False)
        print("trainset: ", len(train_set))
if do_eval:
    with open(devset_dir, "w", encoding="utf-8") as file2:
        dev_set = []
        for j in range(len(dev_input["inputs"])):
            task = dev_input["task"][j]
            dataset = dev_input["dataset"][j]
            instruction = dev_input["instructions"][j]
            input = dev_input["inputs"][j]
            completion = dev_input["labels"][j]
            user_input = "Task: " + task + "\nDataset: ": dataset + "\n" + instruction + "Text: " + input + "\nAnswer: "
            if (len(user_input) + len(completion)) > max_seq_len - 5:
                continue
            dev_set.append({"Task": task, "Dataset": dataset, "user_input": user_input, "completion": completion})
        json.dump(dev_set, file2, ensure_ascii=False) 
        print("devset: ", len(dev_set))
if do_predict:
    with open(testset_dir, "w", encoding="utf-8") as file3:
        test_set = []
        for k in range(len(test_input["inputs"])):
            task = test_input["task"][k]
            dataset = test_input["dataset"][k]
            instruction = test_input["instructions"][k]
            input = test_input["inputs"][k]
            completion = test_input["labels"][k]
            user_input = "Task: " + task + "\nDataset: ": dataset + "\n" + instruction + "Text: " + input + "\nAnswer: "
            if (len(user_input) + len(completion)) > max_seq_len - 5:
                continue
            test_set.append({"Task": task, "Dataset": dataset, "user_input": user_input, "completion": completion})
        json.dump(test_set, file3, ensure_ascii=False) 
        print("testset: ", len(test_set))
