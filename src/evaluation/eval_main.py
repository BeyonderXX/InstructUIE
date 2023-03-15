# 本代码已不再维护，请使用evaluator.py
import json
import re

__all__ = [
    "eval"
]


def eval(json_data, y_preds, task: str):
    """
    integrated evaluation entry
    
    json_data: json-like object
    y_pred: list of strings
    task: {'NER', 'RE', 'Event', 'MRC', 'SM'}
    """
    func_dict = {
        'NER': eval_NER,
        'RE': eval_RE,
        'Event': eval_Event,
        'MRC': eval_MRC,
        'SM': eval_SM
    }
    metric_dict = {
        'NER': 'f1',
        'RE': 'f1',
        'Event': 'f1',
        'MRC': 'acc',
        'SM': 'acc'
    }
    assert task in func_dict, 'Invalid task name %s.'%(task)

    metric = metric_dict[task]

    if metric == 'acc':
        scores = []
        for json_datus, y_pred in zip(json_data, y_preds):
            score = func_dict[task](json_datus, y_pred)
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == 'f1':
        # f1 score 应该在数据集集合尺度上计算 precision、recall，而不是对单个样本计算，之后求平均
        sum_TP = sum_FN = sum_FP = 0
        for json_datus, y_pred in zip(json_data, y_preds):
            TP, FN, FP = func_dict[task](json_datus, y_pred)
            sum_TP += TP
            sum_FN += FN
            sum_FP += FP
        return _calc_f1_score(sum_TP, sum_FN, sum_FP)

def _remove_redundant_space(s):
    # '   a  b  \t  c  \n' --> 'a b c'
    return ' '.join(s.split())

def _TFPN(truth: set, pred: set):
    # TP: 在truth中存在，且在pred中存在
    # FN: 在truth中存在，但在pred中不存在
    # FP: 在truth中不存在，但在pred中存在
    TP = len(truth & pred)
    FN = len(truth - pred)
    FP = len(pred - truth)
    return TP, FN, FP

def _calc_f1_score(TP, FN, FP):
    # TP + FN 可能为0
    # TP + FP 可能为0
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return f1


def eval_NER(json_data, y_pred: str):
    """
    json_data: json-like object
    y_pred: single string outputed by model
    return: TP, FN, FP
    """
    entity_truth = set()
    tmp = json.loads(json_data['instance']['entities'].replace("'",'"'))

    for ent in tmp:
        ent = ent['type']+':'+ent['name']
        ent = _remove_redundant_space(ent)
        entity_truth.add(ent)

    entity_pred = set()
    for ent in y_pred.split(','):
        ent = ent.split(':')
        if len(ent) == 2:
            # 预测中不符合格式的实体会被抛弃
            ent = ent[0].strip()+':'+ent[1].strip()
            ent = _remove_redundant_space(ent)
            entity_pred.add(ent)
    
    return _TFPN(entity_truth, entity_pred)

def eval_RE(json_data, y_pred: str):
    """
    json_data: json-like object
    y_pred: single string outputed by model
    return: TP, FN, FP
    """
    truth = set()
    for rel in json.loads(json_data['Instance']['entities']):
        head = rel['head']['name']
        rel_type = rel['type']
        tail = rel['tail']['name']
        if rel_type not in ['no_relation', 'NA', 'N/A']:
            # 数据集中type为'no_relation'或'NA'的关系会被全部忽略掉
            rel = '(%s,%s,%s)'%(head, rel_type, tail)
            rel = _remove_redundant_space(rel)
            truth.add(rel)

    # y_pred中理应只包含模型自己输出的内容，而不包含prompt
    y_pred = y_pred.split('Answer:')[-1]

    pred = set()
    if y_pred.strip() != 'no relation':
        # 如果模型输出'no relation'，则认为其预测的关系集合为空集
        for rel in re.findall(r'\(.+?\)', y_pred):
            rel = rel[1:-1]
            elements = tuple([i.strip() for i in rel.split(',')])
            if len(elements) == 3:
                # 预测中不符合格式的关系会被舍弃
                rel_type = elements[1]
                if rel_type not in ['no_relation', 'NA', 'N/A']:
                    rel = '(%s,%s,%s)'%elements
                    rel = _remove_redundant_space(rel)
                    pred.add(rel)
    return _TFPN(truth, pred)


def eval_MRC(json_data, y_pred: str):
    """
    json_data: json-like object
    y_pred: single string outputed by model
    return: right or not
    """
    y = _remove_redundant_space(json_data['answer_text'])
    y_pred = _remove_redundant_space(y_pred)
    if y == y_pred:
        return 1
    else:
        return 0


def eval_SM(json_data, y_pred: str):
    """
    data: json-like object
    y_pred: single string outputed by model
    return: right or not
    """
    y = _remove_redundant_space(json_data['label'])
    y_pred = _remove_redundant_space(y_pred)
    trans_dict = {
        '是': 'Yes',
        '否': 'No',
        'yes': 'Yes',
        'no': 'No'
    }
    if y in trans_dict:
        y = trans_dict[y]
    if y_pred in trans_dict:
        y_pred = trans_dict[y_pred]

    if y_pred == y:
        return 1
    else:
        return 0


def eval_Event(json_data, y_pred: str):
    """
    json_data: json-like object
    y_pred: single string outputed by model
    return: TP, FN, FP
    """
    truth = set()
    for event in json_data['events']:
        event_elements = []

        event_type = event['type']
        event_elements.append('event type:%s'%event_type.strip())
        
        trigger = event['trigger']
        event_elements.append('trigger:%s'%trigger.strip())

        for arg in event['arguments']:
            name = arg['name']
            role = arg['role']
            event_elements.append('%s:%s'%(role.strip(), name.strip()))
        
        event_string = ','.join(sorted(event_elements)) # 'a:b,c:d'
        event_string = _remove_redundant_space(event_string)
        truth.add(event_string)
    
    pred = set()
    for event in re.findall(r'\(.+?\)', y_pred):
        event = event[1:-1]
        valid_event = True
        pair_strings = []
        for pair in event.split(','):
            pair = pair.split(':')
            if len(pair) != 2:
                valid_event = False
                break
            pair_strings.append('%s:%s'%(pair[0].strip(), pair[1].strip()))
        if valid_event:
            event_string = ','.join(sorted(pair_strings))
            event_string = _remove_redundant_space(event_string)
            pred.add(event_string)
    return _TFPN(truth, pred)


if __name__ == '__main__':
    print(
        eval(
            json.loads("""
            [{
                "sentence": "我在复旦大学上学，想去墨脱喝奶茶。",
                "entities": [
                    {
                        "name": "复旦大学",
                        "type": "organization",
                        "pos": [
                            2,
                            6
                        ]
                    },
                    {
                        "name": "墨脱",
                        "type": "location",
                        "pos": [
                            11,
                            13
                        ]
                    }
                ]
            }]
            """),
            ["organization: 复旦大学, location: 墨脱"],
            task = 'NER'
        ),
        'Expected Result: 1.0',
        eval(
            json.loads("""
            [{
                "sentence": "我在复旦大学上学，想去墨脱喝奶茶。",
                "relations": [
                    {
                        "head": 
                                {"name": "复旦大学",
                                    "type": "organization",
                                    "pos": [
                                        2,
                                        6
                                    ]
                                },
                        "type":
                                "no_relation",
                        "tail":
                                {"name": "墨脱",
                                "type": "location",
                                "pos": [
                                    11,
                                    13
                                    ]
                                }
                    }
                ]
            },
            {
                "sentence": "小明在上海的复旦大学上学。",
                "relations": [
                    {
                        "head": 
                                {"name": "复旦大学",
                                    "type": "organization",
                                    "pos": [
                                        6,
                                        10
                                    ]
                                },
                        "type":
                                "locate in",
                        "tail":
                                {"name": "上海",
                                "type": "location",
                                "pos": [
                                    3,
                                    5
                                    ]
                                }
                    },
                    {
                        "head": 
                                {"name": "小明",
                                    "type": "person",
                                    "pos": [
                                        0,
                                        2
                                    ]
                                },
                        "type":
                                "belong to",
                        "tail":
                                {"name": "复旦大学",
                                "type": "organization",
                                "pos": [
                                    6,
                                    10
                                    ]
                                }
                    }
                ]
            }
            ]
            """),
            ['no relation', '(复旦大学, locate  in, 上海), (李田所,belong to, 复旦大学)'],
            task = 'RE'
        ),
        'Expected Result: 0.5',
        eval(
            json.loads("""
            [{
                "sentence": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
                "events": [
                    {
                        "trigger": "裁员",
                        "type": "组织关系-裁员", 
                        "pos":[
                            2,
                            3
                        ],
                        "arguments": [
                            {
                                "name": "雀巢",
                                "role": "裁员方", 
                                "pos":[
                                    0,
                                    2
                                ]
                            }, 
                            {
                                "name": "4000人",
                                "role": "裁员人数", 
                                "pos":[
                                    4,
                                    9
                                ]
                            }
                        ]
                    }
                ]
            }]
            """),
            ["(event type : 组织关系-裁员, trigger: 裁员, 裁员方: 雀巢, 裁员人数: 4000人), (event type : 组织关系-裁员, trigger: 裁员, 裁员方: 雀巢, 裁员人数: 3000人)"],
            task = 'Event'
        ),
        'Expected Result: 2/3',
        eval(
            json.loads("""
            [{
                "paragraph": "Another approach to brain function is to examine the consequences of damage to specific brain areas. Even though it is protected by the skull and meninges, surrounded by cerebrospinal fluid, and isolated from the bloodstream by the blood\u2013brain barrier, the delicate nature of the brain makes it vulnerable to numerous diseases and several types of damage. In humans, the effects of strokes and other types of brain damage have been a key source of information about brain function. Because there is no ability to experimentally control the nature of the damage, however, this information is often difficult to interpret. In animal studies, most commonly involving rats, it is possible to use electrodes or locally injected chemicals to produce precise patterns of damage and then examine the consequences for behavior.",
                "question": "What sare the benifts of the blood brain barrir?",
                "answer_start": 195,
                "answer_text": "isolated from the bloodstream"
            }]
            """),
            ["isolated from the bloodstream"],
            task = 'MRC'
        ),
        'Expected Result: 1.0',
        eval(
            json.loads("""
            [{
                "sen1": "A man with a hard hat is dancing.",
                "sen2": "A man wearing a hard hat is dancing.",
                "label": "Yes"
            },
            {
                "sen1": "蚂蚁借呗等额还款可以换成先息后本吗",
                "sen2": "借呗有先息到期还本吗",
                "label": "否"
            }
            ]
            """),
            ["是", "no"],
            task = 'SM'
        ),
        'Expected Result: 1.0',
        sep='\n'
    )
