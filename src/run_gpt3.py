import glob
import json
import openai
import tqdm
import os
import random
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field

openai.api_key=os.environ["openai_key"]

@dataclass
class GPT3Arguments(DataTrainingArguments):
    data_dir: str = field(
        default="data/splits", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    output_dir: str = field(
        default="output/gpt3/", metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    gpt3_temprature: float = field(
        default=0, metadata={"help": "the temprature of GPT3."}
    )
    gpt3_top_p: float = field(
        default=1, metadata={"help": "the top_p parameter of GPT3."}
    )
    engine: str = field(
        default="text-davinci-001", metadata={"help": "the openai GPT3 engine to use."}
    )
    


if __name__ == "__main__":
    random.seed(123)
    parser = HfArgumentParser((GPT3Arguments,))
    args, = parser.parse_args_into_dataclasses()
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

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "gpt3_run_config.json"), "w") as fout:
        json.dump(args.__dict__, fout)

    existing_requests = {}
    if os.path.exists(os.path.join(args.output_dir, "predicted_examples.jsonl")):
        with open(os.path.join(args.output_dir, "predicted_examples.jsonl")) as fin:
            for line in fin:
                request_info = json.loads(line)
                existing_requests[request_info["gpt3_input"]] = request_info["gpt3_response"]

    with open(os.path.join(args.output_dir, "predicted_examples.jsonl"), "w") as fout:
        for example in tqdm.tqdm(raw_datasets["test"]):
            encoded_example = data_collator([example])
            example["gpt3_input"] = encoded_example["inputs"][0].strip()
            example["gpt3_target"] = encoded_example["labels"][0].strip()
            if example["gpt3_input"] in existing_requests:
                response = existing_requests[example["gpt3_input"]]
            else:
                response = openai.Completion.create(
                    engine=args.engine,
                    prompt=example["gpt3_input"],
                    temperature=args.gpt3_temprature,
                    max_tokens=args.max_target_length,
                    top_p=args.gpt3_top_p,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            example["gpt3_response"] = response
            # Note: we cut the generated text at the first period, since the GPT3 language model sometimes generates more than one sentences.
            # Our results show that this won't affect the instruct-GPT3 model very much, but will significantly improve the original GPT3 LM.
            example["prediction"] = example["gpt3_response"]["choices"][0]["text"].strip().split(".")[0]
            fout.write(json.dumps(example) + "\n")

        