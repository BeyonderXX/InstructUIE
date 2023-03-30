1. 文件说明

├── alpaca_instructions  
│   ├── alpaca_instructions.json   *// stanford alpaca使用的5w条指令. 已被转为chatllama所接受的格式*  

├── chatllama  
│   ├── rlhf  
│   ├── actor.py                   *// chatllama有监督训练模块的主要逻辑*  
│   └── llama_model.py             *// chatllama修改过的llama模型*  

├── chatllama_configs  
│   ├── config_uie.yaml            *// 执行main.py时的参数设置, 包含llama模型及分词器路径, 检查点保存路径, 数据集路径, lr, bs, epoch等参数*  
│   └── ds_config_llama.json       *// 执行main.py时, 启用deepspeed以数据并行的形式加速模型训练的必要设置*  

├── 7B  
│   ├── checkpoints                *// 检查点将默认保存到该目录下*  
│   ├── consolidated.00.pth        *// 权重文件*  
│   └── tokenizer.model            *// 分词器*  

├── llama                          *// 原开源代码的模型实现部分. 由于将权重合并了来实现多卡上数据并行, fairyscale张量并行计算模块均改为普通线性层*  

├── UIE_dataset                    *// generate_dataset.py在处理完数据集后, 将转为chatllama格式的数据集默认保存在该文件夹中*  

├── IE_data_v3                     *// 第3版本的数据集, 原始格式, 包括NER, RE, EE*  

├── v3_configs                     *// generate_dataset.py中load_dataset的参数, 决定哪些数据集将用于训练, 验证或测试中*  

├── concatenate_sharps.py          *// 将多个权重文件合并的代码, 逻辑是在相应维度上简单拼接*

├── generate_dataset.py            *// 将IE_data_v3中的数据集转化为chatllama所接受的格式, 保存在UIE_dataset中*  

├── inference.py                   *// 模型训练完成之后的批量推理代码. 尚未实现多卡并行, 但是可以手动多卡并行*  

├── prompt.json                    *// 所用到的指令*

└── UIE_dataset.py                 *// 数据集+指令统一格式逻辑, 用于generate_dataset.py的load_data函数*  
  

  
2. 使用说明  

step1. pip install requirment.txt  

step2. 远程下载数据集和模型. (需在fdn校内或连vpn)
主机名: 10.176.50.48, 端口号: 49153  
数据集路径: /root/chatllana/IE_data_v3  
模型路径: /root/MODELS/7B  
  
step3. 运行generate.py, 修改参数  
参数说明均在代码中有详细注释. 输出的数据集格式应为:  
{"Task": .., "Dataset": .., "user_input": .., "completion": ..}  

step4. 修改chatllama_configs/config_uie.yaml和chatllama_configs/ds_config_llama.json  
运行CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main.py chatllama_configs/config_uie.yaml --type ACTOR
1. config_uie.yaml中, 只需要修改actor_config部分的参数 (line28)
2. config_uie.yaml中, 可能需要修改的参数为:  
model_path: 你保存的7B文件夹(llama模型权重及分词器)的路径  
checkpoint_folder: 训练好的权重保存路径, 请确保该目录存在. 若检测到不存在, 不会自动生成文件夹而是报错  
tokenizer_folder: tokenizer.model的路径. 虽然写作folder但实际上是文件路径而不是目录路径  
train_dataset_path: 你将generate.py处理过的训练集存放至的位置
dev_dataset_path: 你将generate.py处理过的验证集存放至的位置  
batch_size: 这个要和ds_config_llama.json中的train_micro_batch_size_per_gpu保持一致. 7B建议bs(per gpu)=8, 13B建议bs=16  
epochs: 训练的轮数  
deepspeed_config_path: 你的ds_config_llama.json所在的位置
3. 由于使用deepspeed, 所以学习率实际上在ds_config_llama.json中设置  
4. config_uie.yaml与ds_config_llama.json的初始参数设置, 参考了stanford alpaca指令微调的超参设置, 即: epoch=3, lr=2e-5, bs=128. 在用alpaca_instructions.json对llama指令微调时，请尽量不要改变超参数
  
step5. 训练完成后, 运行inference.py. 具体操作在代码中有详细注释  
  
