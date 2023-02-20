import os
import json
import tqdm
import random
from transformers import HfArgumentParser
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from dataclasses import dataclass, field
from nltk import sent_tokenize


@dataclass
class HeuristicsArguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits/default/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default="data/tasks/", metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    output_dir: str = field(
        default="predictions/default/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    method: str = field(
        default="copy_demo", metadata={"help": "The baseline method, including copy_demo or copy_input."}
    )

if __name__ == "__main__":
    random.seed(123)
    parser = HfArgumentParser((HeuristicsArguments,))
    args, = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py",
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            if args.method == "copy_demo":
                example["prediction"] = example["Positive Examples"][0]["output"]
            elif args.method == "copy_input":
                # first_sent = sent_tokenize(example["Instance"]["input"])[0]
                # predictions.append(first_sent)
                example["prediction"] = example["Instance"]["input"]
            else:
                raise NotImplementedError(f"Method {args.method} is not implemented.")
            fout.write(json.dumps(example) + "\n")