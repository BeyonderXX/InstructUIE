#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from typing import Optional
from argparse import ArgumentParser

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint

from uie_collator import DataCollatorForUIE
from uie_trainer import UIETrainer, DenserEvalCallback, skip_instructions


from uie_dist_dataset import UIEInstructDataset


logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def get_args():
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The directory for saving the UIE train/dev/test splits."
    )
    parser.add_argument(
        "--processed_out_dir",
        default=None,
        type=str,
        help="The directory for saving processed data."
    )
    parser.add_argument(
        "--task_config_dir",
        default=None,
        type=str,
        help="The json file for config training and testing tasks"
    )
    parser.add_argument(
        "--instruction_file",
        default=None,
        type=str,
        help="The instruction file for different tasks."
    )
    parser.add_argument(
        "--instruction_strategy",
        default='single',
        type=str,
        help="How many different instructions to use? Support 'single' and 'multiple' mode."
    )
    parser.add_argument(
        "--max_num_instances_per_task",
        default=10000,
        type=int,
        help="The maximum number of instances we will consider for each training task."
    )
    parser.add_argument(
        "--max_num_instances_per_eval_task",
        default=200,
        type=int,
        help="The maximum number of instances we will consider for each validation/test task."
    )
    parser.add_argument(
        "--num_examples",
        default=0,
        type=int,
        help="number of in-context positive examples."
    )
    parser.add_argument(
        "--add_task_name",
        default=False,
        type=bool,
        help="whether to preappend task name before the task input."
    )
    parser.add_argument(
        "--add_dataset_name",
        default=False,
        type=bool,
        help="whether to preappend dataset name before the task input."
    )
    parser.add_argument(
        "--over_sampling",
        default=False,
        type=bool,
        help="Whether to over sampling the dataset to max_num_instances_per_task"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for sampling the dataset."
    )
    return parser.parse_args()


def save_json_lines(data, file_path):
    with open(file_path, 'w+', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} lines to {file_path}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    args = get_args()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Set seed before initializing model.
    set_seed(args.seed)

    # Get the UIE dataset
    raw_datasets = UIEInstructDataset(
        data_dir=args.data_dir,
        instruction_file=args.instruction_file,
        instruction_strategy=args.instruction_strategy,
        task_config_dir=args.task_config_dir,
        num_examples=args.num_examples,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
        over_sampling=args.over_sampling,
        add_task_name=args.add_task_name,
        add_dataset_name=args.add_dataset_name
    )
    if not os.path.exists(args.processed_out_dir):
        os.makedirs(args.processed_out_dir)

    save_json_lines(raw_datasets.train_data()[0], os.path.join(args.processed_out_dir, 'train_.jsonl'))
    save_json_lines(raw_datasets.dev_data()[0], os.path.join(args.processed_out_dir, 'dev_.jsonl'))
    save_json_lines(raw_datasets.test_data()[0], os.path.join(args.processed_out_dir, 'test_.jsonl'))
    save_json_lines(raw_datasets.train_data()[1], os.path.join(args.processed_out_dir, 'train.jsonl'))
    save_json_lines(raw_datasets.dev_data()[1], os.path.join(args.processed_out_dir, 'dev.jsonl'))
    save_json_lines(raw_datasets.test_data()[1], os.path.join(args.processed_out_dir, 'test.jsonl'))


if __name__ == "__main__":
    main()
