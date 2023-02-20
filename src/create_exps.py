import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_experiment.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/mosaic-cirrascale"
# cluster = "ai2/aristo-cirrascale"
# cluster = "ai2/danielk-a100-cluster-50"
# cluster = "ai2/danielk-a100-cluster-30-preemtible"
cluster = "ai2/yizhongw-v100-cluster-5"

def set_argument_value(arguments, name, value):
    if name not in arguments:
        raise ValueError(f"{name} not in arguments.")
    idx = arguments.index(name)
    assert not (isinstance(arguments[idx+1], str) and arguments[idx+1].startswith("-")) # make sure the next argument is the value
    arguments[idx+1] = value
    return arguments

# modify here for different set of experiments
experiment_group = "eval_pretrained_models"

encodings = {
    # "input_only": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "task_name_input": {"add_task_name": True, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "pos_1_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 1, "num_neg_examples": 0, "add_explanation": False},
    # "pos_2_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
    # "pos_4_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 4, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_1_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 1, "num_neg_examples": 0, "add_explanation": False},
    "instruct_pos_2_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_4_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 4, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_2_neg_2_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
    # "instruct_pos_2_neg_2_explanation_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": True},
    # "tk_instruct": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False, "tk_instruct": True},
}

#--------------- experiments about number of supervision tasks -------------------------

if experiment_group == "num_of_tasks":
    train_task_nums = [8, 32, 128, 256, 512]
    for train_task_num in train_task_nums:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{train_task_num}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/train_{train_task_num}")
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- experiments about instances per task -------------------------

if experiment_group == "num_of_instances":
    instance_per_task_nums = [8, 32, 64, 128, 256, 512]
    for num_instance_per_task in instance_per_task_nums:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{num_instance_per_task}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", num_instance_per_task)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- experiments about model variants -------------------------
if experiment_group == "model":
    model_names = [
        # "t5-3b", 
        # "t5-11b",
        # "t5-small", 
        # "t5-large", 
        # "t5-base", 
        # "google/t5-v1_1-small", 
        # "google/t5-v1_1-base", 
        # "google/t5-v1_1-large",
        # "google/t5-v1_1-xl",
        "google/t5-xl-lm-adapt",
        # "google/t5-xxl-lm-adapt",
        # "google/t5-small-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-base-lm-adapt",
        ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        if "small" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
        elif "base" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
        elif "large" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
        elif "3b" in model_name or "-xl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 2)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 8
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- experiments about learning rate and batch size -------------------------
if experiment_group == "hyper_tuning":
    learning_rates = [1e-5, 3e-5, 5e-5, 1e-4, 1e-3]
    acc_steps = [2, 4, 8, 16]
    for lr in learning_rates:
        for acc_step in acc_steps:

            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_training_lr_{lr}_accu_{acc_step}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--learning_rate", lr)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", acc_step)
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)

            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)

# --------------- experiments about the encodings of NI elements -------------------------
if experiment_group == "encoding":
    for encoding_name, encoding in encodings.items():
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_t0_subset_{experiment_group}_{encoding_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
        set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
        set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
        set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
        set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
        if "tk_instruct" in encoding:
            set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- different test sets -------------------------
if experiment_group == "test_sets":

    for set_idx in range(10):
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_test_set_{set_idx}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/set_{set_idx}")

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- different splits -------------------------
if experiment_group == "splits":

    for split_name in ["default", "no_synthetic", "supervised"]:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{split_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/{split_name}")

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)


#--------------- no-finetuning transfer of pretrained models -------------------------
if experiment_group == "eval_pretrained_models":
    model_names = [
        # "google/t5-xl-lm-adapt",
        # "google/t5-xxl-lm-adapt",
        # "bigscience/T0",
        # "bigscience/T0_3B",
        # "t5-large",
        # "google/t5-large-lm-adapt",
        # "google/mt5-xxl",
        "allenai/tk-instruct-11b-def-pos",
        # "allenai/tk-instruct-3b-def-pos", 
        # "allenai/mtk-instruct-3b-def-pos", 
    ]

    for model_name in model_names:
        for encoding_name, encoding in encodings.items():
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_evaluation_model_{model_name.split('/')[-1]}_encoding_{encoding_name}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            assert d['tasks'][0]['command'][3].endswith(".py")
            # d['tasks'][0]['command'] = ["python"] + d['tasks'][0]['command'][3:] 
            d['tasks'][0]['command'].remove("--do_train")
            d['tasks'][0]['command'].remove("--bf16")
            # d['tasks'][0]['command'].remove("--deepspeed")
            # d['tasks'][0]['command'].remove("ds_configs/stage2.config")

            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--disable_tqdm", False)

            set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
            set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
            set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
            set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
            set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", "no")

            if "tk_instruct" in encoding:
                set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])
            
            
            # set model and resources
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
            d['tasks'][0]['resources']['gpuCount'] = 1
            if "small" in model_names:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            elif "base" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            elif "large" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            elif "3b" in model_name or "3B" in model_name or "-xl" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 4)
            elif "11b" in model_name or "11B" in model_name or "-xxl" in model_name or model_name == "bigscience/T0":
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 2)
                d['tasks'][0]['resources']['gpuCount'] = 8
                set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)

#--------------- evaluation of beaker checkpoints -------------------------
if experiment_group == "eval_ckpt":
    checkpoints = [
        # checkpoint_name, beaker_dataset, checkpoint (if None, will use the root beaker output dir)
        # ("input_only", "01FZHYPTKGEN16404MV5TTMJAK", None),
        # ("task_name_input", "01FZHYPV1CNJFNDGNWDJ84G2XV", None),
        # ("instruct_input", "01FZHYPS9XC47R5CZA7RN8TTPQ", None),
        # ("pos_1_input", "01FZHYPSR0Z8S7F901ZTK2SDER", None),
        # ("instruct_pos_1_input", "01FZHYPSZ4SPGV47R0GVR1JFDT", None),
        # ("pos_2_input", "01FZHYPTTE4YGQEXECSPBZGA28", None),
        # ("instruct_pos_2_input", "01FZHYPSGWBER9A9B2NV8TXE4Q", None),
        # ("instruct_pos_2_neg_2_input", "01FZHYPT60T8R7GVF4V5FKSK5E", None),
        # ("instruct_pos_2_neg_2_explanation_input", "01FZHYPS30B5J8P3P8MVDFZJSY", None),
        # ("pos_4_input", "01FZHYPV88MHTW3SHGSTZ753E6", None),
        # ("instruct_pos_4_input", "01FZHYPTCTA9XRD45KKQBTBDZ0", None),   
        # ("tk_instruct", "01FZK3EKQPZNCY30KFECK49YZN", "checkpoint-5000"),
        ("mt5-xl", "01G0A8CYHZF5VV2SW3V10Y9CZT", None)
    ]

    for checkpoint_name, beaker_dataset_id, checkpoint_step in checkpoints:
        for encoding_name, encoding in encodings.items():
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_{experiment_group}_{checkpoint_name}_test_encoding_{encoding_name}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            assert d['tasks'][0]['command'][3].endswith(".py")
            d['tasks'][0]['command'] = ["python"] + d['tasks'][0]['command'][3:] 
            d['tasks'][0]['command'].remove("--do_train")
            d['tasks'][0]['command'].remove("--bf16")
            d['tasks'][0]['command'].remove("--deepspeed")
            d['tasks'][0]['command'].remove("ds_configs/stage2.config")

            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            # set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--disable_tqdm", False)

            set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
            set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
            set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
            set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
            set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", "no")

            if "tk_instruct" in encoding:
                set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

            d['tasks'][0]['datasets'].append({"mountPath": "/models/", "source": {"beaker": beaker_dataset_id}})            
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", "/models/" + (checkpoint_step if checkpoint_step else ""))            

            d['tasks'][0]['resources']['gpuCount'] = 1
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 4)

            set_argument_value(d['tasks'][0]['command'], "--run_name", name) 
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)


#--------------- supervised upper bound -------------------------
if experiment_group == "supervised":
    for encoding_name, encoding in encodings.items():
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_supervised_upper_bound_encoding_{encoding_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
        set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
        set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
        set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
        set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
        if "tk_instruct" in encoding:
            set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/supervised/multilingual")
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 1000)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True) 


#--------------- multilingual -------------------------
if experiment_group == "multilingual":
    model_names = [
        "google/mt5-xl",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_supervised_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        if "small" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
        elif "base" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
        elif "large" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
        elif "3b" in model_name or "-xl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 2)
            d['tasks'][0]['resources']['gpuCount'] = 8
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/supervised/multilingual/")
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 1000)
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        
        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)


