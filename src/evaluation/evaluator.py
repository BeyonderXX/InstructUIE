# Auth: Xia Han 
# Date: 2023/03/09

import json
import re

class MetricBase:
    def __init__(self):
        raise NotImplementedError()
    def update(self, y_truth, y_pred):
        raise NotImplementedError()
    def get_metric(self):
        raise NotImplementedError()
    def get_last(self):
        raise NotImplementedError()

class MetricAcc(MetricBase):
    def __init__(self):
        self.scores = []
    def update(self, y_truth: str, y_pred: str):
        if y_truth == y_pred:
            self.scores.append(1)
        else:
            self.scores.append(0)
    def get_metric(self):
        if len(self.scores) == 0:
            return 0
        else:
            return sum(self.scores) / len(self.scores)
    def get_last(self):
        return self.scores[-1]
            
class MetricF1(MetricBase):
    def __init__(self):
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.last_TP = None
        self.last_FN = None
        self.last_FP = None
    def update(self, y_truth: set, y_pred: set):
        # TP: 在truth中存在，且在pred中存在
        # FN: 在truth中存在，但在pred中不存在
        # FP: 在truth中不存在，但在pred中存在
        self.last_TP = len(y_truth & y_pred)
        self.last_FN = len(y_truth - y_pred)
        self.last_FP = len(y_pred - y_truth)
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP
    def get_metric(self):
        # TP + FN 可能为0
        # TP + FP 可能为0
        TP = self.sum_TP
        FN = self.sum_FN
        FP = self.sum_FP
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
    def get_last(self):
        return self.last_TP, self.last_FN, self.last_FP

class MetricF1NA(MetricF1):
    "对于RE中关系类型为NA的特殊处理"
    def update(self, y_truth: set, y_pred: set):
        for truth in y_truth:
            if ',NA,' in truth:
                pattern = re.escape(truth).replace(',NA,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for pred in y_pred:
                    match = pattern.match(pred)
                    if match is not None and match.group(1) != 'NA':     # truth: (A,NA,B); pred:(A,notNA,B)
                        pred_fail = True
                        break
                if not pred_fail:       # 只有当预测中没有给出错误的明确肯定时才加TP
                    self.sum_TP += 1    # 至于FP会在后面统计，这里就不用计算了，否则会造成重复统计
            else:
                if truth in y_pred:
                    self.sum_TP += 1
                else:
                    self.sum_FN += 1
        for pred in y_pred:
            if ',NA,' in pred:
                pattern = re.escape(pred).replace(',NA,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for truth in y_truth:
                    match = pattern.match(truth)
                    if match is not None and match.group(1) != 'NA':    # pred: (A,NA,B); truth:(A,notNA,B)
                        pred_fail = True
                        break
                if pred_fail:
                    self.sum_FP += 1
                else:
                    self.sum_TP += 0    # 这里不太确定，对于pred给出的(A,NA,B)，如果truth中不包含(A,*,B)，是否应该算作TP? 我姑且认为是不算的，预测中给出(A,NA,B)的话，对了不加分，错了要扣分。
            else:
                if pred not in y_truth:
                    self.sum_FP += 1

class AuditBase:
    def __init__(self, record_limit=16):
        # record_limit: maximum size of record, `-1` for infinite, `0` for no record
        self.record_limit = record_limit
        self.cnt = 0
        self.record = []
    def _check(self, last) -> bool:
        # must be overrided
        # return whether be recorded or not
        raise NotImplementedError()
    def update(self, last):
        if self._check(last):
            self.cnt += 1
            if self.record_limit < 0 or len(self.record) < self.record_limit:
                # record limit check
                self.record.append({
                    'json_data': last['json_data'],
                    'predict': last['predict'],
                    'y_truth': list(last['y_truth']),
                    'y_pred': list(last['y_pred'])
                })
    def get_cnt(self):
        return self.cnt
    def get_record(self):
        return self.record
    def get_report(self):
        return {
            'count': self.cnt,
            'record': self.record
        }
    def get_name(self):
        # 默认为类名，如果想要定制名字的话请考虑重载此方法
        return self.__class__.__name__

class AuditVoid(AuditBase):
    "检测空输出"
    def _check(self, last) -> bool:
        return last['predict'].strip() == ''

class AuditLong(AuditBase):
    "检测过长的输出"
    def _check(self, last) -> bool:
        return len(last['predict']) >= 512     # 长度上限根据需要自行修改

class AuditNA(AuditBase):
    "检测包含类型为NA的输出，目前只用于RE"
    def _check(self, last) -> bool:
        for i in last['y_pred']:    # assert isinstance(i, str)
            if ',NA,' in i:
                return True
        return False

class AuditInvalid(AuditBase):
    "检测包含非法标签类型的输出，目前只用于RE"
    def _check(self, last) -> bool:
        valid_labels = re.findall('Option:(.+?)\n', last['json_data']['instruction'])
        if len(valid_labels) == 0:
            # 如果是没有提供option，则忽略该审计项
            return False
        valid_labels = set(valid_labels[0].strip().split(','))

        for pred in last['y_pred']:
            label = pred.split(',')[1]
            if label not in valid_labels:
                return True
        return False

class AuditRepeat(AuditBase):
    "检测复读机"
    def _check(self, last) -> bool:
        pattern = r'(\w{5,})\1{2,}'  # 匹配连续出现三次及以上的长度大于5的子串
        match = re.search(pattern, last['predict'])
        return match is not None

class AuditRetard(AuditBase):
    "检测零分"
    def _check(self, last) -> bool:
        last_metric = last['metric']
        if hasattr(last_metric, 'last_TP'):
            return last_metric.last_TP == 0
        else:
            return False

class EvaluatorBase:
    def __init__(self):
        self.last = dict()
        self._init_audit()
        self._init_metric()
    
    def _init_metric(self):
        # must be overrided to init self.metric
        self.metric = MetricBase()

    def _init_audit(self):
        # override if necessary
        # 如果需要添加其他审计项目或者自定义实例化的话请考虑重载此方法
        self.audit = [
            AuditVoid(),
            AuditLong(),
            AuditRepeat(),
            AuditRetard()
        ]
    
    def _update_audit(self):
        # override if necessary
        for audit in self.audit:
            audit.update(self.last)

    def _extract(self, json_data, predict: str):
        # must be overrided
        # return: y_truth, y_pred
        raise NotImplementedError()

    def add(self, json_data, predict):
        """
        可以输入多条数据，也可以输入一条数据。
        当输入单条数据时，json_data应该为json-like object(即json.load解析出来的)，predict应该为单个字符串。
        当输入为多条数据时，json_data和predict都应该为列表。
        """
        assert isinstance(json_data, list) == isinstance(predict, list)

        if isinstance(json_data, list) and isinstance(predict, list):
            for i, j in zip(json_data, predict):
                self.add(i, j)
                return
        
        # add single case
        y_truth, y_pred = self._extract(json_data, predict)
        self.metric.update(y_truth, y_pred)

        # audit
        # last中存储需要提交审计的所有可能会用到的信息
        self.last['json_data'] = json_data
        self.last['predict'] = predict
        self.last['y_truth'] = y_truth
        self.last['y_pred'] = y_pred
        self.last['metric'] = self.metric

        self._update_audit()
    
    def get_metric(self) -> float:
        return self.metric.get_metric()

    def get_audit_report(self):
        '获取所有审计项结果报告'
        return {
            a.get_name() : a.get_report()
            for a in self.audit
        }
    def dump_audit_report(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_report(), f, indent=4)

    @staticmethod
    def _remove_redundant_space(s):
        # '   a  b  \t  c  \n' --> 'a b c'
        return ' '.join(s.split())

class EvaluatorNER(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict):
        predict = predict.split('Answer:')[-1]

        entity_truth = set()
        tmp = json.loads(json_data['Instance']['entities'].replace("'",'"'))

        for ent in tmp:
            ent = ent['type']+':'+ent['name']
            ent = self._remove_redundant_space(ent)
            entity_truth.add(ent)
        
        entity_pred = set()
        for ent in predict.split(','):
            ent = ent.split(':')
            if len(ent) == 2:
                # 预测中不符合格式的实体会被抛弃
                ent = ent[0].strip()+':'+ent[1].strip()
                ent = self._remove_redundant_space(ent)
                entity_pred.add(ent)
        return entity_truth, entity_pred

class EvaluatorRE(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1NA()  # 对NA类关系的特殊处理

    def _init_audit(self):
        super()._init_audit()    
        self.audit += [
            AuditInvalid(),
            AuditNA()
        ]

    def _extract(self, json_data, predict):
        # predict中理应只包含模型自己输出的内容，而不包含prompt，但这里还是处理一下
        predict = predict.split('Answer:')[-1]

        y_truth = set()
        for rel in json.loads(json_data['Instance']['entities']):
            head = rel['head']['name']
            rel_type = rel['type']
            tail = rel['tail']['name']
            # type为'no_relation'或'NA'的关系现在不忽略，下同
            rel = '(%s,%s,%s)'%(head, rel_type, tail)
            rel = self._remove_redundant_space(rel)
            y_truth.add(rel)

        y_pred = set()
        if predict.strip() not in {'no relation', '[]'}:
            # 如果模型输出'no relation'或'[]'，则认为其预测的关系集合为空集
            for rel in re.findall(r'\(.+?\)', predict):
                rel = rel[1:-1]
                elements = tuple([i.strip() for i in rel.split(',')])
                if len(elements) == 3:
                    # 预测中不符合格式的关系会被舍弃
                    rel_type = elements[1]
                    rel = '(%s,%s,%s)'%elements
                    rel = self._remove_redundant_space(rel)
                    y_pred.add(rel)
        return y_truth, y_pred

class EvaluatorMRC(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict):
        predict = predict.split('Answer:')[-1]

        truth = self._remove_redundant_space(json_data['answer_text'])
        pred = self._remove_redundant_space(predict)
        return truth, pred


class EvaluatorSM(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
        predict = predict.split('Answer:')[-1]

        y_truth = self._remove_redundant_space(json_data['label'])
        y_pred = self._remove_redundant_space(predict)
        trans_dict = {
            '是': 'Yes',
            '否': 'No',
            'yes': 'Yes',
            'no': 'No'
        }
        if y_truth in trans_dict:
            y_truth = trans_dict[y_truth]
        if y_pred in trans_dict:
            y_pred = trans_dict[y_pred]
        return y_truth, y_pred

class EvaluatorEvent(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
        predict = predict.split('Answer:')[-1]

        y_truth = set()
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
            event_string = self._remove_redundant_space(event_string)
            y_truth.add(event_string)
        
        y_pred = set()
        for event in re.findall(r'\(.+?\)', predict):
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
                event_string = self._remove_redundant_space(event_string)
                y_pred.add(event_string)
        return y_truth, y_pred

if __name__ == '__main__':
    eval_ner = EvaluatorNER()
    eval_re = EvaluatorRE()
    eval_event = EvaluatorEvent()
    eval_mrc = EvaluatorMRC()
    eval_sm = EvaluatorSM()
    def test(evaluator:EvaluatorBase, json_str, predict):
        json_data = json.loads(json_str)
        evaluator.add(json_data, predict)
        print(evaluator.get_metric())
        print(evaluator.get_record)
    
    # 因为后来的实际格式与最初表格中的不同，因此下列测试可能无法通过，仅作为示例s
    test(
        eval_ner,
        """
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
        """,
        ["organization: 复旦大学, location: 墨脱"]
    )
    print('Expected Result: 1.0')

    test(
        eval_re,
        """
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
        """,
        ['no relation', '(复旦大学, locate  in, 上海), (李田所,belong to, 复旦大学)']
    )
    print('Expected Result: 0.5')

    test(
        eval_event,
        """
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
        """,
        ["(event type : 组织关系-裁员, trigger: 裁员, 裁员方: 雀巢, 裁员人数: 4000人), (event type : 组织关系-裁员, trigger: 裁员, 裁员方: 雀巢, 裁员人数: 3000人)"]
    )
    print('Expected Result: 2/3')
    test(
        eval_mrc,
        """
        [{
            "paragraph": "Another approach to brain function is to examine the consequences of damage to specific brain areas. Even though it is protected by the skull and meninges, surrounded by cerebrospinal fluid, and isolated from the bloodstream by the blood\u2013brain barrier, the delicate nature of the brain makes it vulnerable to numerous diseases and several types of damage. In humans, the effects of strokes and other types of brain damage have been a key source of information about brain function. Because there is no ability to experimentally control the nature of the damage, however, this information is often difficult to interpret. In animal studies, most commonly involving rats, it is possible to use electrodes or locally injected chemicals to produce precise patterns of damage and then examine the consequences for behavior.",
            "question": "What sare the benifts of the blood brain barrir?",
            "answer_start": 195,
            "answer_text": "isolated from the bloodstream"
        }]
        """,
        ["isolated from the bloodstream"]
    )
    print('Expected Result: 1.0')
    test(
        eval_sm,
        """
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
        """,
        ["是", "no"]
    )
    print('Expected Result: 1.0')