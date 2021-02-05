"""
Usage: Albert Multi-Class Intent Classifier

NOTE !!!!
Existing architecture bug from HuggingFace, forbid use until fixed


NOTE !!!!
Existing architecture bug from HuggingFace, forbid use until fixed


A chinese fine-tune intent classifier from Albert
Based on `Albert` from paper:
    `ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS`
    https://arxiv.org/pdf/1909.11942.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import os.path as opt
import shutil

from typing import List, Text, Any, Optional, Dict, Tuple

from ner_yjcloud.classifiers import INTENT_RANKING_LENGTH
from ner_yjcloud.components import Component
from ner_yjcloud.models.gpu_utils import getGPUs, getAvailabilityGPU

from ner_yjcloud.models.albert import albert_finetune as albert


try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("No torch package found")

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data import TrainingData
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message


class AlbertIntentClassifier(Component):
    """
    Supervised Albert fine-tune model training
    """

    name = "albert_intent_classifier"

    provides = ["intent", "intent_ranking"]

    # requires = ["text_features"]

    defaults = {
        # global training params
        "batch_size": 32,
        "max_seq_len": 64,
        "epochs": 30,
        "walking_epoch_visual": 1,
        "lr": 2e-5,

        # pre-trained model
        "pretrained": "/home/user/yuanyh/ner_dev/libs/ner_yjcloud/models/albert/pretrained",
        "bert_config": "albert_config.json",
        "bert_model": "albert_model.bin",
        "vocab_config": "vocab.txt"
    }

    @classmethod
    def required_packages(cls):  # type: () -> List[Text]
        return ["torch", "transformers"]

    def __init__(self,
                 component_config,
                 model = None,
                 int2idx = None,
                 idx2int = None,
                 ):

        raise ImportError("Existing architecture bug from HuggingFace, forbid use until fixed")

        super().__init__(component_config)
        self.device = self._load_device()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model
        if self.model:
            self.model.to(self.device)

        self.batch_size = self.component_config.get("batch_size", 16)
        self.max_seq_len = self.component_config.get("max_seq_len", 64)
        self.epochs = self.component_config.get("epochs", 10)
        self.lr = self.component_config.get("lr", 1e-3)

        self.walking_epoch_visual = self.component_config.get("walking_epoch_visual", 1)

        self.pre_path = self.component_config.get("pretrained", "./AlbertModel")
        self.bert_config = self.component_config.get("bert_config", "albert_config.json")
        self.bert_model = self.component_config.get("bert_model", "albert_model.bin")
        self.vocab_config = self.component_config.get("vocab_config", "vocab.txt")

        self.int2idx = int2idx
        self.idx2int = idx2int

        self._check_encode_status()

    @staticmethod
    def _load_device():
        """
        check cuda status and amp accuracy machine status
        would use all gpu if set gpu available

        TODO:
        support use fraction of gpu assigned
        """

        gpus = getGPUs()
        gpus = getAvailabilityGPU(gpus)

        device = torch.device("cuda:%d" % gpus.id) if gpus != "cpu" else torch.device("cpu")
        device = torch.device("cuda:0")

        return device


    def _check_encode_status(self):
        status = True
        if not opt.exists(self.pre_path):
            logger.error(f"Not find albert pretrained path {self.pre_path}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_config)):
            logger.error(f"Not find albert config file {self.bert_config}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_model)):
            logger.error(f"Not find albert pretrained model {self.bert_model}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.vocab_config)):
            logger.error(f"Not find albert pretrained model {self.bert_model}")
            status = False

        if not status:
            raise FileNotFoundError("Not find a legal albert model path")


    def train(self,
              training_data: TrainingData,
              config: Optional[RasaNLUModelConfig], **kwargs: Any) -> None:

        # tokenizer = albert.load_pretrained_tokenizer(self.pre_path)
        encoder, tokenizer = albert.load_pretrained(self.pre_path,
                                                    self.bert_config,
                                                    self.bert_model)

        data_iter_pack = albert.DataIterPack(message = training_data.training_examples,
                                             tokenizer = tokenizer,
                                             batch_size = self.batch_size,
                                             max_seq_len = self.max_seq_len,
                                             epochs = self.epochs,
                                             walking_epoch_visual = self.walking_epoch_visual,
                                             lr = self.lr,
                                             device = self.device)

        data_iter_pack.processor()
        data_iter_pack.train(encoder)

        self.model = data_iter_pack.model
        self.int2idx = data_iter_pack.int2idx
        self.idx2int = data_iter_pack.idx2int

    def process(self, message: Message, **kwargs) -> None:
        intent = {"name": None, "confidence": 0.}
        intent_ranking = []

        tokenizer = albert.load_pretrained_tokenizer(self.pre_path)

        packer = albert.DataIterPack(message = None,
                                     tokenizer = tokenizer,
                                     max_seq_len = self.max_seq_len,
                                     model = self.model,
                                     int2idx = self.int2idx,
                                     idx2int = self.idx2int,
                                     device = self.device)

        if message.text.strip():
            score, label = packer.decode(message.text, INTENT_RANKING_LENGTH)

            intent = {"name": label[0], "confidence": score[0]}

            intent_ranking = [{"name": x, "confidence": y} for x, y in zip(label[1:], score[1:])]

        message.set("intent", intent, add_to_output = True)
        message.set("intent_ranking", intent_ranking, add_to_output = True)


    @classmethod
    def load(cls,
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional[Component] = None,
             **kwargs: Any) -> Any:

        meta = model_metadata.for_component(cls.name)

        t_path = meta.get("pretrained")

        if model_dir and t_path:
            t_path = opt.join(model_dir, t_path)
            e_path = opt.join(opt.join(t_path, meta.get("intent_dict")))


            encoder, tokenizer = albert.load_pretrained(t_path,
                                                        meta.get("bert_config"),
                                                        meta.get("bert_model"))

            with open(e_path, "rb") as f:
                intent_dict = pickle.load(f)

            model_to_load = torch.load(opt.join(t_path, meta.get("bert_model")))
            model_to_load = model_to_load.module if hasattr(model_to_load, "module") else model_to_load

            # assign weight to fine-tune model
            fine_tune_model = albert.AlbertFineTuneModel(encoder,
                                                         len(intent_dict.get("int2idx")),
                                                         if_training = False)
            fine_tune_model.load_state_dict(model_to_load)

            fine_tune_model.eval()

            meta["pretrained"] = t_path

            return AlbertIntentClassifier(component_config = meta,
                                          model = fine_tune_model,
                                          int2idx = intent_dict.get("int2idx"),
                                          idx2int = intent_dict.get("idx2int"))

    @staticmethod
    def copy_file_to_dir(input_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        shutil.copy2(input_file, output_dir)


    def persist(self, model_dir):
        if self.model is None:
            return {"pretrained": None}

        t_path = opt.join(model_dir, "albert", "albert_model.bin")

        try:
            os.makedirs(opt.dirname(t_path))
        except FileExistsError:
            pass

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), t_path)

        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.bert_config),
                              output_dir = opt.join(model_dir, "albert"))
        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.vocab_config),
                              output_dir = opt.join(model_dir, "albert"))

        e_path = opt.join(model_dir, "albert/intent_dict.pkl")
        with open(e_path, "wb") as f:
            pickle.dump({"int2idx": self.int2idx,
                         "idx2int": self.idx2int},
                        f)


        return {"pretrained": "albert",
                "bert_model": "albert_model.bin",
                "bert_config": "albert_config.json",
                "vocab_config": "vocab.txt",
                "intent_dict": "intent_dict.pkl"}


