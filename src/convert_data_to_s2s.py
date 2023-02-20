'''
This script is used for converting our json data into input/output format and save in tsv file. 
This is used to training the T5-11B model on TPU. 
'''

import os
import json
import glob
import tqdm
import pandas as pd
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
from nltk import sent_tokenize

@dataclass
class CustomizedArguments:
    output_dir: str = field(
        default="data/text2text/", metadata={"help": "The directory for saving splits."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((DataTrainingArguments, CustomizedArguments))
    args, customized_args = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py",
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    data_collator = DataCollatorForNI(
        tokenizer,
        model=None,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True
    )

    os.makedirs(customized_args.output_dir, exist_ok=True)

    for split in ["train", "test"]:
        with open(os.path.join(customized_args.output_dir, f"{split}.tsv"), "w") as fout1, \
            open(os.path.join(customized_args.output_dir, f"{split}_examples.jsonl"), "w") as fout2:
            for example in tqdm.tqdm(raw_datasets[split]):
                encoded_example = data_collator([example])
                fout1.write(
                    " ".join(encoded_example["inputs"][0].split()) + "\t" + " ".join(encoded_example["labels"][0].split()) + "\n"
                )
                example["s2s_input"] = " ".join(encoded_example["inputs"][0].split())
                example["s2s_output"] = " ".join(encoded_example["labels"][0].split())
                fout2.write(json.dumps(example) + "\n")
        