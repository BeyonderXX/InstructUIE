"""
instruction 中任务的标签集合作为选项提供，故每个任务应当提供label文件
"""

import os
import json


def label_collect(task_path, task_collect_fun, filter_path=None, max_labels=30):
    """
    在任务路径下，为每个数据集，生成一个labels.json
    labels.json 包含内容为所有标签的list，以NER为例：
            ['person', 'location', 'organization']

    Args:
        task_path: task files path which contains datasets dir
        task_collect_fun: task specific label collect function
        filter_path: filter datasets with too much labels

    Returns:

    """
    filter_ds_dirs = []

    for dirpath, dirnames, filenames in os.walk(task_path):
        for dirname in dirnames:
            ds_dir = os.path.join(dirpath, dirname)
            labels = []

            # 假设全部包含train、dev、test数据集
            for ds_type in ['train', 'dev', 'test']:
                ds_type_file = os.path.join(ds_dir, ds_type+'.json')
                ds_type_labels = task_collect_fun(ds_type_file)
                labels += ds_type_labels

            labels = list(set(labels))
            out_file = os.path.join(ds_dir, 'labels.json')
            json.dump(labels, open(out_file, 'w+', encoding='utf-8'), ensure_ascii=False)
            print('Finish out {} labels to {}!'.format(len(labels), out_file))

            if len(labels) >= max_labels:
                filter_dir = os.path.join(filter_path, dirname)
                filter_ds_dirs.append([ds_dir, filter_dir])

    # filter datasets by mv them to another dirs
    for ds_dir, filter_dir in filter_ds_dirs:
        cmd = "mv {} {}".format(ds_dir, filter_dir)
        os.system(cmd)
        print('Move {} to {} cz too much labels!'.format(ds_dir, filter_dir))


def NER_label_collect(file_path):
    """
    按照输入文件路径，收集标签，按照list返回。
    输入文件为一个样例json列表，返回为标签集合
    Args:
        file_path: dataset file name

    Returns:
        [label1, label2]
    """
    fi = open(file_path, 'r+', encoding='utf-8')
    samples = json.load(fi)
    labels = []

    for sample in samples:
        for entity in sample['entities']:
            if entity['type'] not in labels:
                labels.append(entity['type'])

    return labels


if __name__ == "__main__":
    NER_path = '/root/InstructUIE/data/NER_processed'
    filter_path = '/root/InstructUIE/data/NER_filter'
    label_collect(NER_path, NER_label_collect, filter_path)
