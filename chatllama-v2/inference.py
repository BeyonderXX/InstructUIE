# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# 执行命令样例：
# CUDA_VISIBLE_DEVICES=1  python llama/inference_batch_multiGPU.py  "MODELS/llama_7B"  "MODELS/llama_7B/tokenizer.model"  "MODELS/llama_7B/predictions"  1  0  10000  32
'''
参数说明
使用的GPU编号 
python 程序名 (arg0) 
模型权重文件夹 (arg1) 
词嵌入模型路径 (arg2) 
输出文件文件夹 (arg3) 
输出文件编号 (arg4, !!! 需要和GPU编号一起修改 !!!)
处理的第一条数据在数据集的位置 (arg5)
处理的最后一条数据在数据集的位置 (arg6)
batch size (arg7, 默认为32)
**llama源码中设置了max_bs=32(line44), 没有试过修改后能否正常运行
'''
# 以上所有的路径名均填写 !!! 绝对路径 !!!
# 每开一个新进程, 记得设置新的device index, p_index(prediction_file index), from/to_index
# 如:
# (pid1) -> CUDA_VISIBLE_DEVICES=0  python chatllama-v2/inference.py  "chatllama-v2/7B"  "chatllama-v2/7B/tokenizer.model"  "chatllama-v2/7B/predictions"  0  0  10000  32
# (pid2) -> CUDA_VISIBLE_DEVICES=1  python chatllama-v2/inference.py  "chatllama-v2/7B"  "chatllama-v2/7B/tokenizer.model"  "chatllama-v2/7B/predictions"  1  10000  20000  32
# (pid3) -> CUDA_VISIBLE_DEVICES=2  python chatllama-v2/inference.py  "chatllama-v2/7B"  "chatllama-v2/7B/tokenizer.model"  "chatllama-v2/7B/predictions"  2  20000  30000  32
# ...

import os
import sys
import torch
import time
import json
import math
import jsonlines
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(ckpt_dir: str, tokenizer_path: str) -> LLaMA:
    start_time = time.time()
    ckpt_path = Path(ckpt_dir) / "checkpoints/llama-7B.pt"
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    for k,v in checkpoint.items(): # delete the key "model"
        if k == "model":
            checkpoint = v
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=32, **params) 
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(p_index: int, ckpt_dir: str, tokenizer_path: str, output_path: str, from_index: int, to_index: int, batch_size: int, temperature: float = 0.8, top_p: float = 0.95):
    with open("chatllama-v2/UIE_dataset/UIE_test.json", "r", encoding="utf-8") as file: # 数据集绝对路径
        dataset = json.load(file)
    print(len(dataset)) # 可以先打印看看总数，然后除以GPU数，用来填下一行中每个进程处理的数量
    dataset = dataset[from_index: to_index] 
    prompts = []
    labels = []
    tasks = []
    datasets = []
    for item in dataset:
        prompts.append("Task: " + item["Task"] + "\nDataset: " + item["Dataset"] + "\n" + item["user_input"] + "\t")
        labels.append(item["completion"])
        tasks.append(item["Task"])
        datasets.append(item["Dataset"])

    n_batch = math.ceil(len(prompts) / batch_size)

    with jsonlines.open(output_path + "/" + str(p_index) + ".jsonl", "w") as file:
        generator = load(ckpt_dir, tokenizer_path)
        
        for i in range(n_batch):
            start_time = time.time()
            prompts_input = prompts[i * batch_size : min(len(prompts), (i + 1) * batch_size)]
            labels_input = labels[i * batch_size : min(len(prompts), (i + 1) * batch_size)]
            tasks_input = tasks[i * batch_size : min(len(prompts), (i + 1) * batch_size)]
            datasets_input = datasets[i * batch_size : min(len(prompts), (i + 1) * batch_size)]
            results = generator.generate(prompts_input, max_gen_len=1024, temperature=temperature, top_p=top_p)
            for j in range(len(results)):
                split_point = results[j].index("\t")+1
                label = labels_input[j]
                prediction = results[j][split_point:]
                if label == "[]":
                    label = "None"
                if prediction == "[]":
                    prediction = "None"
                file.write({"Task": tasks_input[j], "Dataset":datasets_input[j], "Instance":{"sentence": results[j][:split_point],"label": label}, "Prediction": prediction})
                #file.write("\n==================================\n")
            print(f"One batch done. Inference within {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=x -> p_index=x
    main(p_index=sys.argv[4], ckpt_dir=sys.argv[1], tokenizer_path=sys.argv[2], output_path=sys.argv[3], from_index=sys.argv[5], to_index=sys.arg[6], batch_size=sys.argv[7])
