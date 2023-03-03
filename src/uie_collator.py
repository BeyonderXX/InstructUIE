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

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            # task definition 何时加载的？
            # Done, 目前在loader中加载json

            # add the input first.
            task_input = instance["instruction"]
            task_input = task_input.format(instance['Instance']['sentence'])

            # TODO，方便模型识别数据集
            # task_name = ""
            # if add_task_name:
            #     task_name += instance["Task"] + ". "

            # TODO，测试，原论文结论貌似没用
            # definition = ""
            # if add_task_definition:
            #     if isinstance(instance["Definition"], list):
            #         definition = "Definition: " + instance["Definition"][0].strip()  # TODO: should we use <Definition>?
            #     else:
            #         definition = "Definition: " + instance["Definition"].strip()
            #     if not definition[-1] in string.punctuation:
            #         definition += "."
            #     definition += "\n\n"

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

        if "entities" in batch[0]["Instance"] and batch[0]["Instance"]["entities"]:
            # Randomly select one reference if multiple are provided.
            # 生成，NLU不需要
            # print([ex["Instance"]["entities"] for ex in batch])
            jsons = [json.loads(ex["Instance"]["entities"].replace("'",'"')) for ex in batch]

            labels = []
            for entities in jsons:
                kv_pairs = [[entity['type'], entity['name']] for entity in entities]
                kv_string = ", ".join(["{}: {}".format(k, v) for (k, v) in kv_pairs])
                labels.append(kv_string)

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



# "Please list all entity words in the text that fit the category.Output format is [{word1, type1}, {word2, type2}]
# text: {x}
# category: [{type：location}, {type: org}, ... , ]
# Answer: [{word1, type1}, {word2, type2}, ..., {wordN,typeN}]"

# TODO list
# 1.输出验证
# 2.输入增加labels，先在数据集中生成labels
# 3.save ckpt 设置
# 4.evaluation 连调
# 5.增加任务定义，二阶段优化
