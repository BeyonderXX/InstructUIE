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
        self.last_TP = 0
        self.last_FN = 0
        self.last_FP = 0
        for truth in y_truth:
            if ',na,' in truth:
                pattern = re.escape(truth).replace(',na,', ',(.+),')    # 因为在evaluator._extract的时候就全部变为了小写，所以是na而非NA
                pattern = re.compile(pattern)
                pred_fail = False
                for pred in y_pred:
                    match = pattern.match(pred)
                    if match is not None and match.group(1) != 'na':     # truth: (A,NA,B); pred:(A,notNA,B)
                        pred_fail = True
                        break
                if not pred_fail:       # 只有当预测中没有给出错误的明确肯定时才加TP
                    self.last_TP += 1    # 至于FP会在后面统计，这里就不用计算了，否则会造成重复统计
            else:
                if truth in y_pred:
                    self.last_TP += 1
                else:
                    self.last_FN += 1
        for pred in y_pred:
            if ',na,' in pred:
                pattern = re.escape(pred).replace(',na,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for truth in y_truth:
                    match = pattern.match(truth)
                    if match is not None and match.group(1) != 'na':    # pred: (A,NA,B); truth:(A,notNA,B)
                        pred_fail = True
                        break
                if pred_fail:
                    self.last_FP += 1
                else:
                    self.last_TP += 0    # 这里不太确定，对于pred给出的(A,NA,B)，如果truth中不包含(A,*,B)，是否应该算作TP? 我姑且认为是不算的，预测中给出(A,NA,B)的话，对了不加分，错了要扣分。
            else:
                if pred not in y_truth:
                    self.last_FP += 1
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP

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

class AuditInsane(AuditBase):
    "检测胡言乱语"
    def _check(self, last) -> bool:
        return last['predict'].strip().lower() not in {'na', 'no relation', 'none', '[]', ''} and len(last['y_pred']) == 0    # 说了点什么，但又什么有用的都没说

class AuditBothEmpty(AuditBase):
    "检测Label和predict都为空的条目"
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) == 0

class AuditNA(AuditBase):
    "检测包含类型为NA的输出，目前只用于RE"
    def _check(self, last) -> bool:
        for i in last['y_pred']:    # assert isinstance(i, str)
            if ',na,' in i:
                return True
        return False

class AuditInvalid(AuditBase):
    "检测包含非法标签类型的输出，目前只用于RE"
    def _check(self, last) -> bool:
        valid_labels = re.findall('Option:(.+?)\n', last['json_data']['Instance']['instruction'])
        if len(valid_labels) == 0:
            # 如果是没有提供option，则忽略该审计项
            return False
        valid_labels = set(valid_labels[0].strip().split(','))

        for pred in last['y_pred']:
            pred = pred.split(',')
            if len(pred) == 3:
                label = pred[1]
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
        
class AuditWhatever(AuditBase):
    "无差别逮捕"
    def _check(self, last) -> bool:
        return True

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
            AuditBothEmpty(),
            AuditLong(),
            AuditInsane(),
            AuditRepeat(),
            AuditRetard(),
            AuditWhatever()
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
        #'  kjc,  jns , ((  : ()  )  (  )( ln  kc  a,,  ' --> 'kjc,jns,((:())()(ln kc a,,'
        s = ' '.join(s.split())
        s = re.sub(r'\s*(,|:|\(|\))\s*', r'\1', s)
        return s
    
    @staticmethod
    def _format(s):
        # 集大成的格式规范化，集中解决各种格式的疑难杂症
        s = EvaluatorBase._remove_redundant_space(s)
        s = s.lower()
        s = s.replace('{','').replace('}','')
        s = re.sub(',+', ',', s)
        s.replace('orgnization', 'organization')
        return s
    
    @staticmethod
    def _re_item(s):
        # '   A,B,C),   (D,EF),  ,,(GH ' --> ['A,B,C', 'D,EF', 'GH']
        # ' A,B,C)  ' --> ['A,B,C']
        # 因为有时模型的输出会缺少开头的左括号或者结尾的右括号
        # 该正则表达式不捕获括号，只捕获中间的内容
        return re.findall(r'(?:^|\()([^\(\)]+?)(?:$|\))', s.strip())
    
    @staticmethod
    def _resolve_brackets(s):
        # 将最上层的配对括号内的内容抽取出来，以字符串列表的形式返回，抛弃括号外的内容。
        # 此函数容忍句子开头缺失的一个左括号和句子结尾缺失的一个右括号（但不会同时容忍）
        # 'a(b)(c(d))(
        ans = []
        level = 0
        last_lb_idx = None
        for idx, char in enumerate(s):
            if char == '(':
                if level == 0:
                    last_lb_idx = idx
                level += 1
            elif char == ')':
                if last_lb_idx is None and len(ans) == 0 and 0 != idx:
                    ans.append(s[0 : idx])
                if level == 1 and last_lb_idx+1 != idx:
                    ans.append(s[last_lb_idx+1 : idx])
                if level >= 1:
                    level -= 1
        if level == 1 and last_lb_idx+1 != len(s):
            ans.append(s[last_lb_idx+1:])
        return ans
    
    @staticmethod
    def _resolve_comma(s):
        # 将句子按逗号分割，但是括号内的逗号不算，分割出来的空字符串忽略
        # 'a,(b,c),,d,' --> ['a', '(b,c)', 'd']
        ans = []
        level = 0
        last_comma = -1
        for idx, char in enumerate(s):
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            elif char == ',' and level == 0 and last_comma + 1 != idx:
                ans.append(s[last_comma+1 : idx])
                last_comma = idx
        if last_comma+1 != len(s):
            ans.append(s[last_comma+1:])
        return ans

class EvaluatorNER(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict):
        entity_truth = set()
        for ent in self._resolve_brackets(json_data['Instance']['label']):   # FIXME:字段名可能有变
            ent = self._format(ent)
            entity_truth.add(ent)
        
        entity_pred = set()
        for ent in self._resolve_brackets(predict):
            # 部分地名可能会包含逗号，因此这里不检查逗号个数
            ent = self._format(ent)
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
        y_truth = set()
        for rel in self._resolve_brackets(json_data['Instance']['label']):   # FIXME:字段名可能有变
            # type为'no_relation'或'NA'的关系现在不忽略，下同
            rel = self._format(rel)
            y_truth.add(rel)

        y_pred = set()
        # 如果模型输出'no relation'或'[]'，则认为其预测的关系集合为空集，但这里并不需要做特殊判别
        for rel in self._resolve_brackets(predict):
            # 因为字段中可能本身就存在逗号，此处不再进行数量校验
            rel = self._format(rel)
            y_pred.add(rel)
        return y_truth, y_pred

class EvaluatorMRC(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict):
        truth = self._remove_redundant_space(json_data['answer_text'])
        pred = self._remove_redundant_space(predict)
        return truth.lower(), pred.lower()


class EvaluatorSM(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
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
        return y_truth.lower(), y_pred.lower()

class EvaluatorEvent(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricAcc()
    def _extract(self, json_data, predict):
        y_truth = set()
        for event in self._resolve_brackets(json_data['Instance']['label']):   # FIXME:字段名可能有变
            event = self._format(event)
            event.replace('arguments:', '')
            event_elements = self._resolve_comma(event)  # 因为后面会排序，所以每个pair的规整化需要提前进行
            
            event_string = ','.join(sorted(event_elements)) # 'a:b,c:d'
            y_truth.add(event_string)
        
        y_pred = set()
        for event in self._resolve_brackets(predict):
            event = self._format(event)
            event.replace('arguments:', '')
            event_elements = self._resolve_comma(event)  # 因为后面会排序，所以每个pair的规整化需要提前进行
            
            event_string = ','.join(sorted(event_elements)) # 'a:b,c:d'
            y_truth.add(event_string)
        return y_truth, y_pred

# 因为后来的实际格式与最初表格中的不同，因此下列测试可能无法通过，仅作为使用示例
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