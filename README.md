# InstructUIE

- This repo releases our implementation for the InstructUIE model.
- It is built based on the pretrained [T5 model](https://arxiv.org/abs/1910.10683), and finetuned on our data.

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can install the required libraries by running 

```bash
bash setup.sh
```


## Data

Our models are trained and evaluated on [InstructUIE data](https://github.com/allenai/natural-instructions), which can be cloned by running:

```bash
TODO
```

If you want to use the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer), you can convert the data into text2text format with [`scripts/convert_data_to_s2s.sh`](scripts/convert_data_to_s2s.sh).

## Training

A sample script for training the Tk-Instruct 3B model in our paper can be found at [`scripts/train_tk_instruct.sh`](scripts/train_tk_instruct.sh). You can run it as follows:

```bash
./scripts/train_uie_instruct.sh
```


## Released Checkpoints

Our 3B and 11B model checkpoints are accessible via the [Hugging Face Hub](https://huggingface.co/models?search=tk-instruct-). You can load them easily using the [Transformers](https://github.com/huggingface/transformers) library:

```python
TODO
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def")

>>> input_ids = tokenizer.encode(
        "Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
        return_tensors="pt"
    )
>>> output = model.generate(input_ids, max_length=10)
>>> output = tokenizer.decode(output[0], skip_special_tokens=True)
```

## Evaluation

The following script evaluates our 3B Tk-Instruct model that uses `task definition + 2 positive examples` as instructions:

```bash
TODO
```

This should give you a ROUGE-L score of ~54.0, as is reported in the Table 3 of our [paper](https://arxiv.org/pdf/2204.07705.pdf).

You can also try other models under different encodings. You can control whether to include definition / explanation, or the number of pos/neg examples, by specifying the arguments in [`src/run_s2s.py`](src/run_s2s.py).


## Model Predictions and Performance

TODO
The predictions of our tested models can be found in the [`output`](output/) folder. You can evaluate each predition file in the following way:

```bash
python src/compute_metrics.py --predictions output/default/tk-instruct-3b-def-pos/predicted_examples.jsonl --track default --compute_per_category_metrics
python src/compute_metrics.py --predictions output/xlingual/mtk-instruct-3b-def-pos/predicted_examples.jsonl --track xlingual --compute_per_category_metrics
```

Here are the performance numbers (in ROUGE-L) for our tested models:

|                          | Models                  | Default Track (en) | X-lingual Track |
|--------------------------|-------------------------|--------------------|-----------------|
| Heuristic Baselines      | Copying Instance Input  | 14.20              | 5.44            |
|                          | Copying Demo. Output    | 28.54              | 50.31           |
| Pretrained LMs           | T5-LM (11B)             | 30.16              | -               |
|                          | GPT3 (175B)             | 45.05              | 51.20           |
| Instruction-tuned Models | T0 (11B)                | 32.28              | -               |
|                          | GPT3-Instruct (175B)    | 52.06              | 53.74           |
|                          | Tk-Instruct (Ours, 3B)  | 54.33              | -               |
|                          | Tk-Instruct (Ours, 11B) | 60.07              | -               |
|                          | mTk-Instruct (Ours, 3B) | -                  | 56.72           |

Note that these numbers might be different from the numbers reported in the our arxiv paper, because we 1) resampled our evaluation instances; 2) updated our evaluation script. We will update the paper once allowed.

We will keep adding the predictions and performance of new models into this repository.

## Citation
TODO


