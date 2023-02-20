import json
import os
import tqdm
import glob

task_instance_to_id_map = {}

for file in glob.glob("output/**/*.jsonl", recursive=True):
    print(file)
    new_predicted_examples = []
    with open(file) as fin:    
        for line in tqdm.tqdm(fin):
            example = json.loads(line)
            task = example["Task"]
            if task not in task_instance_to_id_map:
                with open(os.path.join('data/tasks/', f'{task}.json')) as fin:
                    task_data = json.load(fin)
                instance_to_id_map = {}
                for instance in task_data["Instances"]:
                    assert instance["input"] not in instance_to_id_map
                    instance_to_id_map[instance["input"]] = instance["id"]
                task_instance_to_id_map[task] = instance_to_id_map
            else:
                instance_to_id_map = task_instance_to_id_map[task]
            assert example["Instance"]["input"] in instance_to_id_map
            instance_id = instance_to_id_map[example["Instance"]["input"]]
            example["id"] = instance_id
            example["Instance"]["id"] = instance_id
            new_predicted_examples.append(example)

    with open(file, 'w') as fout:
        for example in new_predicted_examples:
            fout.write(json.dumps(example) + '\n')
            

    