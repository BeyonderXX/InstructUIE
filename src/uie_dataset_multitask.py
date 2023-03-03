# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)


class UIEConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir = None, prompt_dir= None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        if prompt_dir == None:
            print('none')
            prompt_dir =r'/workspace/InstructUIE/prompt.json'
        with open(prompt_dir, 'r') as f:
            self.prompt_dict = json.loads(f.read())


class UIEInstructions(datasets.GeneratorBasedBuilder):


    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = UIEConfig
    BUILDER_CONFIGS = [
        UIEConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"


    # def __int__(self,*args):


    # TODO，不同数据集格式需要统一，暂时先按照NER的加载
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "instruction": datasets.Value("string"),

                    # "Categories": [datasets.Value("string")],
                    # "Definition": [datasets.Value("string")],
                    # "Input_language": [datasets.Value("string")],
                    # "Output_language": [datasets.Value("string")],
                    # "Instruction_language": [datasets.Value("string")],
                    # "Domains": [datasets.Value("string")],
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "sentence": datasets.Value("string"),
                        "entities": datasets.Value("string")
                    }
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            logger.error("Please provide right input: data_dir or task_dir!")

        # 原始代码按照 split dir 去记录数据集信息，分别加载
        # task dir 存放所有数据集，任务按照split分为了训练和测试，对应于unseen task的测试
        # 暂时改为有监督setting，根据所有数据集的训练集训练， 同时在所有数据集的测试集上测试，数据全部放在split dir下
        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
        ]

    # 原始path为train/test文件路径，现在可以不用task_dir，直接在path下加载
    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        assert os.path.exists(path)
        task_dir = task_dir.split(',')
        for task_catagory in task_dir:
            task_path = os.path.join(path, task_catagory)
            for task_file_name in os.listdir(task_path):
                task_file_path_ = os.path.join(task_path, task_file_name)
                task_file_path = os.path.join(task_file_path_, subset + ".json")

                id = 0

                instruction_list = self.config.prompt_dict[task_catagory]
                instruction = random.choice(instruction_list)


                if task_catagory == 'RE':
                    labels_path = os.path.join(task_file_path_, "label.json")

                    with open(labels_path, encoding="utf-8") as labels_f:
                        labels_json = json.load(labels_f)
                        labels_str = ','.join(labels_json)
                        instruction += "Option:" + labels_str + " \n " + "Answer: "

                    with open(task_file_path, encoding="utf-8") as task_f:
                        s = task_f.read()
                        task_data_list = json.loads(s)
                        sample_template = {"Task": task_catagory}
                        sample_template["instruction"] = instruction
                        instances = task_data_list

                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]

                    for idx, instance in enumerate(instances):
                        example = sample_template.copy()
                        example["id"] = str(idx)
                        for k, entity in enumerate(instance['relations']):
                            instance['relations'][k]['head']['name'] = entity['head']['name'].replace("'", "#$%#")
                            instance['relations'][k]['tail']['name'] = entity['tail']['name'].replace("'", "#$%#")
                            instance['relations'][k]['type'] = entity['type'].replace("'", "#$%#")
                        example["Instance"] = {
                            "sentence": instance['sentence'],
                            "entities": json.dumps(instance['relations'])
                        }
                        id += 1

                        yield f"{task_file_name}##{idx}", example

                elif task_catagory == 'NER':
                    labels_path = os.path.join(task_file_path_, "labels.json")

                    with open(labels_path, encoding="utf-8") as labels_f:
                        labels_json = json.load(labels_f)
                        labels_str = ','.join(labels_json)
                        instruction += "Option:" + labels_str + " \n " + "Answer: "

                    with open(task_file_path, encoding="utf-8") as task_f:
                        s = task_f.read()
                        task_data_list = json.loads(s)
                        sample_template = {"Task": task_catagory}
                        sample_template["instruction"] = instruction
                        instances = task_data_list

                        if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                            random.shuffle(instances)
                            instances = instances[:max_num_instances_per_task]

                        for idx, instance in enumerate(instances):
                            example = sample_template.copy()
                            example["id"] = str(idx)
                            for k, entity in enumerate(instance['entities']):
                                instance['entities'][k]['type'] = entity['type'].replace("'", "#$%#")
                                instance['entities'][k]['name'] = entity['name'].replace("'", "#$%#")
                            example["Instance"] = {
                                "sentence": instance['sentence'],
                                "entities": json.dumps(instance['entities'])
                            }
                            id += 1

                            yield f"{task_file_name}##{idx}", example






def load_examples(path=None, task_dir = None , max_num_instances_per_task=None, subset=None):
    """Yields examples."""
    with open(r'D:\pycharmProject\GDUAP\test_dataset\prompt.json', 'r') as f:
        prompt_dict = json.loads(f.read())

    logger.info(f"Generating tasks from = {path}")
    assert os.path.exists(path)

    for task_catagory in task_dir:
        task_path = os.path.join(path, task_catagory)
        for task_file_name in os.listdir(task_path):
            task_file_path_ = os.path.join(task_path, task_file_name)
            task_file_path = os.path.join(task_file_path_, subset + ".json")

            id = 0

            instruction_list = prompt_dict[task_catagory]
            instruction = random.choice(instruction_list)

            if task_catagory == 'RE':
                labels_path = os.path.join(task_file_path_, "label.json")

                with open(labels_path, encoding="utf-8") as labels_f:
                    labels_json = json.load(labels_f)
                    labels_str = ','.join(labels_json)
                    instruction += "Option:" + labels_str + " \n " + "Answer: "

                with open(task_file_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data_list = json.loads(s)
                    sample_template = {"Task": task_catagory}
                    sample_template["instruction"] = instruction
                    instances = task_data_list

                if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                    random.shuffle(instances)
                    instances = instances[:max_num_instances_per_task]

                for idx, instance in enumerate(instances):
                    example = sample_template.copy()
                    example["id"] = idx
                    example["Instance"] = {
                        "sentence": instance['sentence'],
                        "entities": json.dumps(instance['relations'])
                    }
                    id += 1
                    print(task_file_name)
                    yield f"{task_file_name}##{idx}", example

            elif task_catagory == 'NER':
                labels_path = os.path.join(task_file_path_, "labels.json")

                with open(labels_path, encoding="utf-8") as labels_f:
                    labels_json = json.load(labels_f)
                    labels_str = ','.join(labels_json)
                    instruction += "Option:" + labels_str + " \n " + "Answer: "

                with open(task_file_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data_list = json.loads(s)
                    sample_template = {"Task": task_catagory}
                    sample_template["instruction"] = instruction
                    instances = task_data_list

                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]

                    for idx, instance in enumerate(instances):
                        example = sample_template.copy()
                        example["id"] = idx
                        example["Instance"] = {
                            "sentence": instance['sentence'],
                            "entities": json.dumps(instance['entities'])
                        }
                        id += 1
                        print(task_file_name)

                        yield f"{task_file_name}##{idx}", example


if __name__ == "__main__":
    sample_genor = load_examples(path=r'D:\pycharmProject\GDUAP\test_dataset\IE_data',task_dir=['RE','NER'],
                                 max_num_instances_per_task=20, subset='train')

    id = 0
    for sample in sample_genor:
        print(sample)
