import logging
import string
import json

import torch
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
        self.tokenizer.padding_side = 'right'
        if return_tensors is None:
            return_tensors = self.return_tensors

        if 'codegen'.lower() in self.model.config._name_or_path.lower() or 'bloomz'.lower() in self.model.config._name_or_path.lower() :
            #decoder模型处理

            sources = []
            len_list = []

            labels = []

            for instance in batch:
                #加入Instruct  既t5模型里的source
                task_input = instance["instruction"]
                task_input = task_input.format(instance['Instance']['sentence'])

                #处理生成label

                entities = json.loads(instance["Instance"]["entities"].replace("'", '"').replace("#$%#", "'"))
                if entities:
                    kv_pairs = []
                    relation_pairs = []
                    event_pairs = []
                    for entity in entities:
                        # 分别处理NER和RE
                        if 'type' in entity and 'trigger' in entity and 'arguments' in entity:
                            event_type = entity['type']
                            event_trigger = entity['trigger']
                            event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for
                                               argument in entity['arguments']]
                            event_pairs_ = [event_type, event_trigger, event_arguments]
                            event_pairs.append(event_pairs_)
                        elif 'type' in entity and 'name' in entity:
                            kv_pairs_ = [entity['type'], entity['name']]
                            kv_pairs.append(kv_pairs_)

                        elif 'head' in entity and 'type' in entity and 'tail' in entity:
                            relation_pairs_ = [entity['head']['name'], entity['type'], entity['tail']['name']]
                            relation_pairs.append(relation_pairs_)

                    if len(event_pairs) > 0:
                        label = ", ".join(
                            ["(type:{}, trigger:{}, arguments:".format(type, trigger) + ", ".join(arguments) + ")"
                             for (type, trigger, arguments) in event_pairs])
                    elif len(kv_pairs) > 0:
                        label = ", ".join(["{}: {}".format(k, v) for (k, v) in kv_pairs])
                    elif len(relation_pairs) > 0:
                        label = ", ".join(["({}, {}, {})".format(h, r, t) for (h, r, t) in relation_pairs])
                    # labels.append(label)

                else:
                    # labels.append("[]")
                    label = '[]'

                tokenized_source = self.tokenizer(task_input)["input_ids"]
                if len(tokenized_source) <= self.max_source_length:
                    len_list.append(len(tokenized_source))
                else:
                    len_list.append(self.max_source_length)
                    task_input = self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True)

                tokenized_source = self.tokenizer(label)["input_ids"]
                if len(tokenized_source) <= self.max_target_length:
                    pass
                else:
                    label = self.tokenizer.decode(tokenized_source[:self.max_target_length], skip_special_tokens=True)
                labels.append(label)
                sources.append(task_input+label)

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
            loss_mask = torch.ones((model_inputs['input_ids'].shape[0],model_inputs['input_ids'].shape[1]-1))
            for k,instruct_len in enumerate(len_list):
                loss_mask[k,:instruct_len-1] = 0

            model_inputs["labels"] = model_inputs['input_ids']
            model_inputs['loss_mask'] = loss_mask
            return model_inputs
        else:
            # s2s模型处理
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

            if "entities" in batch[0]["Instance"] and batch[0]["Instance"]["entities"]:
                jsons = [json.loads(ex["Instance"]["entities"].replace("'", '"').replace("#$%#", "'")) for ex in batch]

                labels = []
                for entities in jsons:
                    if entities:
                        kv_pairs = []
                        relation_pairs = []
                        event_pairs = []
                        for entity in entities:
                            # 分别处理NER和RE
                            if 'type' in entity and 'trigger' in entity and 'arguments' in entity:
                                event_type = entity['type']
                                event_trigger = entity['trigger']
                                event_arguments = ["(name:{},role:{})".format(argument['name'], argument['role']) for argument
                                                   in entity['arguments']]
                                event_pairs_ = [event_type, event_trigger, event_arguments]
                                event_pairs.append(event_pairs_)
                            elif 'type' in entity and 'name' in entity:
                                kv_pairs_ = [entity['type'], entity['name']]
                                kv_pairs.append(kv_pairs_)

                            elif 'head' in entity and 'type' in entity and 'tail' in entity:
                                relation_pairs_ = [entity['head']['name'], entity['type'], entity['tail']['name']]
                                relation_pairs.append(relation_pairs_)

                        if len(event_pairs) > 0:
                            label = ", ".join(
                                ["(type:{}, trigger:{}, arguments:".format(type, trigger) + ", ".join(arguments) + ")" for
                                 (type, trigger, arguments) in event_pairs])
                        elif len(kv_pairs) > 0:
                            label = ", ".join(["{}: {}".format(k, v) for (k, v) in kv_pairs])
                        elif len(relation_pairs) > 0:
                            label = ", ".join(["({}, {}, {})".format(h, r, t) for (h, r, t) in relation_pairs])
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
