# TODO: amp accurate

import os.path as opt
import os

from typing import Any, List, Tuple, Optional, Dict, Union

from collections import namedtuple, defaultdict
from copy import deepcopy
from tqdm import tqdm
import pickle
import json
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from transformers import BertTokenizer, AlbertModel
from transformers import AlbertConfig as BC
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def load_pretrained(mpath,
                    config = "albert_config.json",
                    model = "albert_model.bin"):

    b_config = BC.from_pretrained(opt.join(mpath, config))
    encoder = AlbertModel.from_pretrained(opt.join(mpath, model), config = b_config)
    tokenizer = BertTokenizer.from_pretrained(mpath)

    return encoder, tokenizer


def load_pretrained_tokenizer(mpath):

    return BertTokenizer.from_pretrained(mpath)


def load_pretrained_encoder(mpath,
                            config = "albert_config.json",
                            model = "albert_model.bin"):

    b_config = BC.from_pretrained(opt.join(mpath, config))
    encoder = AlbertModel.from_pretrained(opt.join(mpath, model), config = b_config)

    return encoder

class InputExample:
    """a single examples object"""
    def __init__(self,
                 idx: Union[int, str],
                 sentence: str,
                 label: Union[int, str]) -> None:
        self.idx = idx
        self.sentence = sentence
        self.label = label


class InputFeature:
    """feature object for single example"""
    def __init__(self,
                 idx: Union[int, str],
                 input_ids: List[int],
                 input_mask: List[int],
                 segment_ids: List[int],
                 label_id: Union[int, str],
                 is_real_example: bool = True) -> None:
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataIterPack():
    """obtain batch sample and features"""

    def __init__(self,
                 message,
                 tokenizer,
                 batch_size = 16,
                 max_seq_len = 64,
                 epochs = 10,
                 walking_epoch_visual = 1,
                 lr = 1e-3,
                 device = torch.device("cpu"),
                 int2idx = None,
                 idx2int = None,
                 model = None):
        self.message = message
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.lr = lr

        self.device = device

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.walking_epoch_visual = walking_epoch_visual

        self.model = model

        self.int2idx, self.idx2int = int2idx, idx2int
        self.examples, self.features = None, None


    def _create_intent_dict(self, intents: List[str]) -> None:
        """create label to idx encoding dictionary"""
        int2idx = defaultdict()

        for intent in intents:
            if intent in int2idx:
                continue
            else:
                int2idx[intent] = len(int2idx)

        idx2int = {idx: val for val, idx in int2idx.items()}

        self.int2idx, self.idx2int = int2idx, idx2int


    def load_content(self, examples: Any) -> List[namedtuple]:

        # for exam in examples:
        #     yield exam.get("text"), exam.get("intent")
        cont_tuple = namedtuple("cont", ["text", "intent"])
        content = []

        for exam in examples:
            content.append(cont_tuple(text = exam.get("text"),
                                      intent = exam.get("intent")))
            # content.append(cont_tuple(text = exam.text,
            #                           intent = exam.intent))

        return content

    def convert_sentence_features(self, sentence):
        """convert single sentence to corresponding features"""
        tokens = self.tokenizer.tokenize(sentence)[:(self.max_seq_len - 2)]
        tokens = ["CLS"] + tokens + ["SEP"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        return input_ids, input_mask, segment_ids


    def convert_feature(self, content):
        """convert raw content to `InputExample` and `InputFeature`"""
        examples = []
        features = []

        for idx, cont in enumerate(content):
            example = InputExample(str(idx),
                                   cont.text,
                                   cont.intent)

            input_ids, input_mask, segment_ids = self.convert_sentence_features(cont.text)
            label_id = self.int2idx.get(cont.intent)

            feature = InputFeature(str(idx),
                                   input_ids,
                                   input_mask,
                                   segment_ids,
                                   label_id)

            examples.append(example)
            features.append(feature)

        self.examples, self.features = examples, features


    def make_batch(self):
        features = deepcopy(self.features)
        b_input_ids = torch.as_tensor([x.input_ids for x in features], device = self.device)
        b_input_mask = torch.as_tensor([x.input_mask for x in features], device = self.device)
        b_segment_ids = torch.as_tensor([x.segment_ids for x in features], device = self.device)
        b_labels = torch.as_tensor([x.label_id for x in features], device = self.device)

        dataset = TensorDataset(b_input_ids, b_input_mask, b_segment_ids, b_labels)

        train_dataloader = DataLoader(dataset,
                                      sampler = RandomSampler(features),
                                      batch_size = self.batch_size)

        return train_dataloader


    def processor(self):
        content = self.load_content(self.message)
        self._create_intent_dict([x.intent for x in content])
        self.convert_feature(content)


    def decode(self, text, ranking):
        self.model.eval()

        input_ids, input_mask, segment_ids = self.convert_sentence_features(text)
        input_ids = torch.as_tensor(input_ids, device = self.device).view(1, -1)
        input_mask = torch.as_tensor(input_mask, device = self.device).view(1, -1)
        segment_ids = torch.as_tensor(segment_ids, device = self.device).view(1, -1)

        logits, _, _ = self.model(input_ids, input_mask, segment_ids)

        pointer = logits.flatten().argsort(descending = True).tolist()[:ranking]
        score = torch.exp(logits.flatten()) / torch.exp(logits.flatten()).sum()
        score = score[pointer].tolist()
        return score, [self.idx2int[x] for x in pointer]


    def train(self, encoder):

        train_dataloader = self.make_batch()

        self.model = AlbertFineTuneModel(encoder,
                                         len(self.int2idx),
                                         if_training = True)
        self.model.to(self.device)

        last_loss = 0.
        train_acc = 0.
        pbar = tqdm(range(self.epochs), desc = "Epoches")

        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)


        for ep in pbar:
            ep_loss = 0.
            self.model.train()

            for idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                # b_params = self.obtain_batch_feature(batch)
                logits, loss, train_acc = self.model(*batch)
                ep_loss += loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()


            if ep % self.walking_epoch_visual == 0:
                # logger.info(f"epoch: {ep}\tLoss: {ep_loss}")
                pbar.set_postfix({"epoch": "%d" % ep,
                                  "loss": f"{ep_loss:.3f}",
                                  "acc": f"{train_acc:.3f}"})
                last_loss = ep_loss


        if self.walking_epoch_visual:
            logger.info("Finished training albert finetune policy, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))



class AlbertFineTuneModel(nn.Module):
    def __init__(self,
                 encoder,
                 num_labels,
                 dropout = 0.2,
                 if_training = True):
        super(AlbertFineTuneModel, self).__init__()
        self.num_labels = num_labels

        self.bert = encoder
        if if_training:
            for param in self.bert.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.loss_func = CrossEntropyLoss(reduction = "mean")


    def call_loss(self, logits, labels):
        loss = self.loss_func(logits.view(-1, self.num_labels),
                              labels.view(-1))

        return loss

    def call_acc(self, logits, labels):
        train_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        return train_acc

    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                labels = None):


        pool_output = self.bert(input_ids = input_ids,
                                token_type_ids = token_type_ids,
                                attention_mask = attention_mask
                                )[1]

        pool_output = self.dropout(pool_output)
        logits = self.classifier(pool_output)
        loss, accuracy = 0., 0.

        if self.training:
            loss = self.call_loss(logits, labels)
            accuracy = self.call_acc(logits, labels)

        return logits, loss, accuracy


class DataSimu():
    def __init__(self, text, intent):
        self.text = text
        self.intent = intent



# if __name__ == '__main__':
#     with open("./output_anno.json") as f:
#         lines = json.load(f)
#
#         lines = lines["rasa_nlu_data"]["common_examples"][:1000]
#
#         examples = [DataSimu(x["text"], x['intent']) for x in lines]
#
#     encoder, tokenizer = load_pretrained('./pretrained')
#
#     trainer = DataIterPack(examples, tokenizer)
#     trainer.processor()
#     trainer.train()
#
#     trainer.persist()







