import logging
import string
import json
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForUIE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False

    # tk_instruct 决定是否使用全部的instruct做训练

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:

            task_input = instance["instruction"]
            task_input = task_input.format(instance['Instance']['sentence'])

            source = task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(
                    self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        # TODO， 修改 key
        if "entities" in batch[0]["Instance"] and batch[0]["Instance"]["entities"]:
            jsons = [json.loads(ex["Instance"]["entities"].replace("'", '"').replace("#$%#", "'")) for ex in batch]

            labels = []
            for entities in jsons:
                if entities:
                    kv_pairs = []
                    relation_pairs = []
                    event_pairs = []
                    # TODO， 针对任务分别封装，仅返回label结果
                    for entity in entities:
                        # 分别处理NER和RE
                        if 'type' in entity and 'trigger' in entity and 'arguments' in entity:
                            event_type = entity['type']
                            event_trigger = entity['trigger']
                            event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for argument in entity['arguments']]
                            event_pairs_ = [event_type,event_trigger,event_arguments]
                            event_pairs.append(event_pairs_)
                        elif 'type' in entity and 'name' in entity:
                            kv_pairs_ = [entity['type'], entity['name']]
                            kv_pairs.append(kv_pairs_)

                        elif 'head' in entity and 'type' in entity and 'tail' in entity:
                            relation_pairs_ = [entity['head']['name'],entity['type'],entity['tail']['name']]
                            relation_pairs.append(relation_pairs_)

                    if len(event_pairs)>0:
                        label = ", ".join(["(type:{}, trigger:{}, arguments:".format(type, trigger)+", ".join(arguments)+")" for (type, trigger, arguments) in event_pairs])
                    elif len(kv_pairs)>0:
                        label = ", ".join(["{}: {}".format(k, v) for (k, v) in kv_pairs])
                    elif len(relation_pairs)>0:
                        label = ", ".join(["({}, {}, {})".format(h,r,t) for (h,r,t) in relation_pairs])
                    labels.append(label)
                else:
                    labels.append("[]")

            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model,
                                              "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs



# TODO list
# 1.输出验证
# 2.输入增加labels，先在数据集中生成labels
# 3.save ckpt 设置
# 4.evaluation 连调
# 5.增加任务定义，二阶段优化
