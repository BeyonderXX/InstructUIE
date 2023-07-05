# InstructUIE

- This repo releases our implementation for the InstructUIE model.
- It is built based on the pretrained Flan T5 model, and finetuned on our data.

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.26.1)
- DeepSpeed (0.7.7)

You can install the required libraries by running 

```
bash setup.sh
```


## Data

Our models are trained and evaluated on **IE INSTRUCTIONS**. 
You can download the data from [Baidu NetDisk](https://pan.baidu.com/s/1R0KqeyjPHrsGcPqsbsh1XA?from=init&pwd=ybkt) or [Google Drive](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view?usp=share_link).


## Training

A sample script for training the InstructUIE model in our paper can be found at [`scripts/train_flan-t5.sh`](scripts/train_flan-t5.sh). You can run it as follows:

```
bash ./scripts/train_flan-t5.sh
```


## Released Checkpoints

We have released our 11B UIE model, click [here](https://huggingface.co/ZWK/InstructUIE) for download.


## Evaluation

A sample script for evaluating the InstructUIE model in our paper can be found at [`scripts/eval_flan-t5.sh`](scripts/eval_flan-t5.sh). You can run it as follows:

```
bash ./scripts/eval_flan-t5.sh
```
The decoded results would save to predict_eval_predictions.jsonl in your output dir. 
To calculate f1 with predict_eval_predictions.jsonl
```
python calculate_f1.py
```




