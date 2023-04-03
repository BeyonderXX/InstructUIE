"""InstructUIE Dataset."""

import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class UIEConfig:
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            instruction_file=None,
            instruction_strategy=None,
            task_config_dir=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            add_task_name=None,
            add_dataset_name=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.instructions = self._parse_instruction(instruction_file)
        self.task_configs = self._parse_task_config(task_config_dir)
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task
        self.add_task_name = add_task_name
        self.add_dataset_name = add_dataset_name

    def _parse_instruction(self, instruction_file):
        """
        Instruction example:
        {
          "RE": [
            {"instruction_type": "zero-shot", "instruction": "Given a phrase that describes the relationship between
            two words, extract the words and the lexical relationship between them.
            The output format should be :[(word1, relation, word2)]. \n"},
          ],
          "NER": [
            {"instruction_type": "zero-shot", "instruction": "Please list all entity words in the text that
            fit the category.Output format is [(word1, type1), (word2, type2))]. \n"},
          ],
          "EE": [
            {"instruction_type": "zero-shot", "instruction": "Extract the event information in the text
            and return them in the event list. \n"}
          ]
        }
        """
        if not instruction_file:
            return None
        instructions = {"zero-shot": {}, "few-shot": {}}

        with open(instruction_file, 'r+') as f:
            origin_instructions = json.load(f)

        for task in origin_instructions:
            for task_instruction in origin_instructions[task]:
                instruct_type = task_instruction["instruction_type"]
                if instruct_type == "zero-shot":
                    instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [])
                    instructions['zero-shot'][task].append(task_instruction["instruction"])
                elif instruct_type == "few-shot":
                    instructions['few-shot'][task] = instructions['few-shot'].get(task, [])
                    instructions['few-shot'][task].append(task_instruction["instruction"])
                else:
                    raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                     .format(instruct_type, instruction_file))
        return instructions

    def _parse_task_config(self, task_config_dir):
        """
        Task config file example:
            {
              "RE": [
                {"sampling strategy": "random", "dataset name": "conll04"}
              ],
              "NER": [
                {"sampling strategy": "random", "dataset name": "ACE05_coarse-grained"},
                {"sampling strategy": "full", "dataset name": "conll2003"}
              ],
              "EE": [
                {"sampling strategy": "random", "dataset name": "GENIA"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs


# TODO, few-shot, 需要 load 的时候就将值存好，放在 "Examples" 里面
class UIEInstructDataset:
    """
        InstructUIE Dataset preprocess for distributed training.


        sample format:
            {
                "Task": datasets.Value("string"),
                "Dataset": datasets.Value("string"),
                "subset": datasets.Value("string"),
                "Samples": [{
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "ground_truth": datasets.Value("string")
                }],
                "Instance": {
                    "input": datasets.Value("string"),
                    "predict": datasets.Value("string"),

                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "ground_truth": datasets.Value("string")
                }
            }
    """

    def __init__(self, *args, **kwargs):
        self._test_data = None
        self._dev_data = None
        self._train_data = None
        self.config = UIEConfig(*args, **kwargs)
        self.load_samples()

    def train_data(self):
        if self._train_data is None:
            raise ValueError("train_data is None, please load_samples first!")
        return self._train_data

    def dev_data(self):
        if self._dev_data is None:
            raise ValueError("dev_data is None, please load_samples first!")
        return self._dev_data

    def test_data(self):
        if self._test_data is None:
            raise ValueError("test_data is None, please load_samples first!")
        return self._test_data

    def load_samples(self):
        """
        Load samples from data_dir, and generate train/dev/test data.
        """
        # default load total test data
        subsets = {"train": self.config.max_num_instances_per_task,
                   "dev": self.config.max_num_instances_per_eval_task,
                   "test": None}
        path = self.config.data_dir

        for subset, max_mun_instances_per_task in subsets.items():
            logger.info("Loading {} data from {})".format(subset, path))
            task_config = self.config.task_configs[subset]
            subset_samples = self._generate_examples(
                path=path,
                task_config=task_config,
                max_num_instances_per_task=max_mun_instances_per_task,
                subset=subset
            )

            if subset == "train":
                self._train_data = subset_samples
            elif subset == "dev":
                self._dev_data = subset_samples
            else:
                self._test_data = subset_samples
            logger.info("Loaded {} {} samples".format(len(subset_samples), subset))

    def _load_dataset(self, dataset_path, labels_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)

        return instances, labels

    def get_instruction(self, sentence, instruction_template, task, dataset, labels_str, num_examples=None):
        # "instructions \n options \n {0} \n Answer: "
        instruction_template += "Option:" + labels_str + " \n" + "Text: " + "{0}" + "\n" + "Answer:"
        content = sentence

        # add task/ds prefix
        prefix = ''
        if self.config.add_task_name:
            prefix += "Task:" + task + '\n'
        if self.config.add_dataset_name:
            prefix = prefix + "Dataset:" + dataset + '\n'

        instruction_template = prefix + instruction_template

        # TODO, add in-context samples
        samples = ''
        if num_examples and num_examples > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content

        # add content
        # TODO, support for slot filling task, e.g. Relation Classification
        instruction = instruction_template.format(content)

        return instruction

    def _get_instruction_template(self, task):
        """
        Get instruction template for task.
        """
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        if self.config.num_examples is not None and self.config.num_examples > 0:
            task_instructions = self.config.instructions['few-shot'][task]
        else:
            task_instructions = self.config.instructions['zero-shot'][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)

    # support over sampling
    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances

    def load_NER_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        # TODO, support few-shot
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        labels_str = ','.join(labels)

        for idx, instance in enumerate(instances):
            instruction_template = self._get_instruction_template('NER')
            instruction = self.get_instruction(instance['sentence'], instruction_template, 'NER', dataset_name, labels_str)
            kv_pairs = []

            for entity in instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)

            if len(kv_pairs) > 0:
                label = ", ".join(["({}, {})".format(k, v) for (k, v) in kv_pairs])
            else:
                label = "None"

            example = {
                "inputs": instruction,
                "targets": label,

                "Task": "NER",
                "Dataset": dataset_name,
                "Samples": [],
                "subset": subset,
                "id": str(idx),
                "sentence": instance['sentence'],
                "ground_truth": label,
                "instruction_template": instruction_template
            }
            model_inputs = {
                "inputs": instruction,
                "targets": label
            }

            yield example, model_inputs

    def load_RE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        labels_str = ','.join(labels)

        for idx, instance in enumerate(instances):
            instruction_template = self._get_instruction_template('RE')
            instruction = self.get_instruction(instance['sentence'], instruction_template, 'RE', dataset_name, labels_str)
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = ", ".join(["({}, {}, {})".format(h, r, t) for (h, r, t) in relation_pairs])
            else:
                label = 'None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ", ".join(["({}, {}, {})".format(h, r, t) for (h, r, t) in ground_truth_pairs])
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))

            example = {
                "inputs": instruction,
                "targets": label,

                "Task": "RE",
                "Dataset": dataset_name,
                "Samples": [],
                "subset": subset,
                "id": str(idx),
                "sentence": instance['sentence'],
                "ground_truth": ground_truth,
                "instruction_template": instruction_template
            }
            model_inputs = {
                "inputs": instruction,
                "targets": label
            }

            yield example, model_inputs

    def load_EE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)

        # TODO, reconstruct Event Instruction to two stage
        labels_str = f'Event type: {labels[0]}, Arguments type: {labels[1]}.'
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            instruction_template = self._get_instruction_template('EE')
            instruction = self.get_instruction(instance['sentence'], instruction_template, 'EE', dataset_name, labels_str)
            event_pairs = []

            for k, event in enumerate(instance['events']):
                if event['type'] == 'NA' or event['type'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for
                                   argument in event['arguments']]

                event_arguments = "None" if not event_arguments else ", ".join(event_arguments)
                event_pair = [event_type, event_trigger, event_arguments]
                event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = ", ".join(["(type:{}, trigger:{}, arguments:{})".format(type, trigger, arguments)
                                   for (type, trigger, arguments) in event_pairs])
            else:
                label = 'None'

            example = {
                "inputs": instruction,
                "targets": label,

                "Task": "EE",
                "Dataset": dataset_name,
                "Samples": [],
                "subset": subset,
                "id": str(idx),
                "sentence": instance['sentence'],
                "ground_truth": label,
                "instruction_template": instruction_template
            }
            model_inputs = {
                "inputs": instruction,
                "targets": label
            }

            yield example, model_inputs

    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        instances = []
        inputs_instances = []

        for task in task_config:
            task_instances = []
            task_inputs = []
            if task == "NER":
                load_func = self.load_NER_dataset
            elif task == 'RE':
                load_func = self.load_RE_dataset
            elif task == 'EE':
                load_func = self.load_EE_dataset
            else:
                raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))

            # load dataset
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                sampling_strategy = dataset.get("sampling strategy", "random")
                ds_path = os.path.join(path, task, ds_name, subset + '.json')
                labels_path = os.path.join(path, task, ds_name, 'labels.json')
                assert os.path.exists(ds_path)
                assert os.path.exists(labels_path)

                idx = -1
                dataset_instances = []
                ds_inputs = []
                for sample, model_inputs in load_func(ds_path, labels_path, ds_name, sampling_strategy,
                                                      max_num_instances_per_task, subset):
                    idx += 1
                    dataset_instances.append(sample)
                    ds_inputs.append(model_inputs)

                task_instances.extend(dataset_instances)
                task_inputs.extend(ds_inputs)
            instances.extend(task_instances)
            inputs_instances.extend(task_inputs)

        return instances, inputs_instances

