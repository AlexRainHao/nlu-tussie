import warnings
warnings.filterwarnings("ignore")

import os
import os.path as opt

from typing import Any, List, Tuple, Optional, Dict, Union

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from ner_tussie.models import clsEvalCounter, report_score_matr

import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset, Subset
from torch.utils.data.dataset import random_split
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig as BC

logger = logging.getLogger(__name__)


def load_pretrained(mpath,
                    config = "bert_config.json",
                    model = "bert_model.bin"):

    """load pre-trained bert encoder and tokenizer"""

    b_config = BC.from_pretrained(opt.join(mpath, config))
    encoder = BertModel.from_pretrained(opt.join(mpath, model), config = b_config)
    tokenizer = BertTokenizer.from_pretrained(mpath)

    return encoder, tokenizer


def load_encoder(mpath, config):
    config = BC.from_pretrained(opt.join(mpath, config))
    return BertModel(config)


def load_pretrained_tokenizer(mpath):
    """load pre-trained tokenizer"""

    return BertTokenizer.from_pretrained(mpath)


def load_pretrained_config(mpath, config = "bert_config.json"):
    return BC.from_pretrained(opt.join(mpath, config))


def load_pretrained_encoder(mpath,
                            config = "bert_config.json",
                            model = "bert_model.bin"):
    """load pre-trained bert encoder"""

    b_config = BC.from_pretrained(opt.join(mpath, config))
    encoder = BertModel.from_pretrained(opt.join(mpath, model), config = b_config)

    return encoder


class IntentDataset():
    """pass"""

    def __init__(self, intent):
        """

        Parameters
        ----------
        intent: List[str]
        """
        self.intent = intent
        self.int2idx = defaultdict(int)

    def encoder(self, label):

        if label in self.int2idx:
            pass
        else:
            self.int2idx[label] = len(self.int2idx)


    def reversed(self):
        return {id_v: int_v for int_v, id_v in self.int2idx.items()}


    def __call__(self):
        for lab in self.intent:
            self.encoder(lab)

        return self.int2idx, self.reversed()



class NluClsDataset(Dataset):
    """pass"""

    def __init__(self, message, tokenizer, max_len, label_dict):
        """

        Parameters
        ----------
        message: Any, NLU input data
        tokenizer: Tokenizer
        max_len: int, assign max sequences length
        label_dict: dict, a intent to idx dictionary
        """
        self.message = message
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = label_dict

    def __len__(self):
        return len(self.message)

    def __getitem__(self, idx):
        feature = self.message[idx].get("text") # str
        label = self.message[idx].get("intent") # str

        encoding = self.tokenizer.encode_plus(
            feature,
            add_special_tokens = True,
            max_length = self.max_len,
            truncation = True,
            return_token_type_ids = False,
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt",
        )

        return {
            "text": feature,
            "input_ids": encoding["input_ids"].flatten(),
            "input_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.label_dict.get(label), dtype = torch.long)
        }


class NluClsDataLoader:
    """pass"""
    def __init__(self, message, tokenizer, max_len, batch_size, label_dict):
        self.dl = self.create_data_loader(message, tokenizer, max_len, label_dict)
        self.batch_size = batch_size


    def create_data_loader(self, message, tokenizer, max_len, label_dict):
        dl = NluClsDataset(
            message = message,
            tokenizer = tokenizer,
            max_len = max_len,
            label_dict = label_dict
        )

        return dl

    def __len__(self):
        return len(self.dl)


    def split_train_test(self, split_ratio):
        """split train and validation for training"""
        dl = deepcopy(self.dl)
        f_size = len(dl)
        train_size = int(split_ratio * f_size)
        test_size = f_size - train_size

        train_dataset, test_dataset = random_split(dl, [train_size,
                                                        test_size])
        return train_dataset, test_dataset


    def refresh(self):
        """refresh for all `Dataset` and convert to new shuffled `DataLoader`"""
        dl = deepcopy(self.dl)
        return DataLoader(dl, self.batch_size, num_workers = 4, shuffle = True)


class TrainingPipeLine():
    """pass"""
    def __init__(self,
                 epochs = 10,
                 walking_epoch_visual = 1,
                 lr = 2e-5,
                 dropout = 0.2,
                 device = torch.device("cpu"),
                 int2idx = None,
                 idx2int = None):

        self.epochs = epochs
        self.walking_epoch_visual = walking_epoch_visual
        self.lr = lr
        self.dropout = dropout
        self.device = device
        self.int2idx = int2idx
        self.idx2int = idx2int
        
        self.eval_res = {}

    def train(self, encoder, data_loader, test_loader = None):
        """
        Parameters
        ----------
        data_loader: NluClsDataLoader: training samples data loader
        test_loader: Union[NluClsDataLoader, None]: test samples data loader
        """

        model = BertFineTuneModel(encoder, len(self.int2idx), self.dropout)
        model.to(self.device)

        pbar = tqdm(range(self.epochs), desc = "Epochs")

        optimizer = AdamW(model.parameters(), lr = self.lr)
        total_steps = len(data_loader.dl) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 0,
                                                    num_training_steps = total_steps)
        ep_loss = 0.
        acc = 0.
        for ep in pbar:
            ep_loss = 0.
            model.train()
            b_data_loader = data_loader.refresh()

            for batch in b_data_loader:
                optimizer.zero_grad()
                b_input_ids = batch["input_ids"].to(self.device)
                b_input_mask = batch["input_mask"].to(self.device)
                b_label = batch["label"].to(self.device)

                logits, loss, acc = model(b_input_ids,
                                          b_input_mask,
                                          b_label)

                ep_loss += loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            if self.walking_epoch_visual and ep % self.walking_epoch_visual == 0:
                # add validation loss
                _, _eval_res = self.evaluate(model, test_loader)
                self.eval_res[ep] = _eval_res

                pbar.set_postfix({"ep": f"{ep}",
                                  "trl": f"{ep_loss:.3f}",
                                  "tra": f"{acc:.3f}"})

        model.eval()
        # evaluation score
        _, _eval_res = self.evaluate(model, test_loader)
        if _eval_res:
            self.eval_res[ep] = _eval_res
            logger.info(report_score_matr(_eval_res))
        logger.info("Finished training albert finetune policy, "
                    "loss={:.3f}, train accuracy={:.3f}"
                    "".format(ep_loss, acc))
        return model


    def evaluate(self, model, test_loader):
        """
        evaluation during training if `test_loader` assigned
        Parameters
        ----------
        test_loader: Union[NluClsDataLoader, None]

        """

        test_loss = 0.
        pred_res = []
        true_res = []

        if not test_loader:
            return test_loss, {}

        model.eval()

        test_data_loader = test_loader.refresh()

        with torch.no_grad():
            for batch in test_data_loader:
                t_input_ids = batch["input_ids"].to(self.device)
                t_input_mask = batch["input_mask"].to(self.device)
                t_label = batch["label"].to(self.device).tolist()

                logits, loss, acc = model(t_input_ids, t_input_mask, t_label)
                t_pred_labels = logits.argmax(dim=-1).tolist()

                pred_res.extend(t_pred_labels)
                true_res.extend(t_label)

                test_loss += loss

        cls_res = clsEvalCounter(self.idx2int).run(true_res, pred_res)

        return test_loss, cls_res


    def decode(self, model, tokenizer, max_len, text, ranks):
        model.eval()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = max_len,
            truncation = True,
            return_token_type_ids = False,
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt",
        )

        input_ids = encoding["input_ids"].view(1, -1).to(self.device)
        input_mask = encoding["attention_mask"].view(1, -1).to(self.device)

        logits, _, _ = model(input_ids, input_mask, None)

        pointer = logits.flatten().argsort(descending = True).tolist()[:ranks]
        score = torch.exp(logits.flatten()) / torch.exp(logits.flatten()).sum()
        score = score[pointer].tolist()

        return score, [self.idx2int[x] for x in pointer]



class BertFineTuneModel(nn.Module):

    def __init__(self, encoder, num_labels, dropout = .2):
        super(BertFineTuneModel, self).__init__()

        self.bert = encoder
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.cls_layer = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_func = CrossEntropyLoss(reduction = "mean")



    def forward(self, input_ids, input_mask, labels = None):
        _, pool_output = self.bert(input_ids = input_ids,
                                   attention_mask = input_mask)

        pool_output = self.dropout(pool_output)
        logits = self.cls_layer(pool_output)

        loss, accuray = 0., 0.

        if self.training:
            loss = self.call_loss(logits, labels)
            accuray = self.call_acc(logits, labels)

        return logits, loss, accuray


    def call_loss(self, logits, labels):
        loss = self.loss_func(logits.view(-1, self.num_labels),
                              labels.view(-1))

        return loss

    def call_acc(self, logits, labels):
        acc = (logits.argmax(dim = -1) == labels).float().mean().item()
        return acc
