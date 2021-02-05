"""
Usage: Bert Multi-Class Intent Classifier

A chinese fine-tune intent classifier from Bert
Based on `Bert` from paper:
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`
    https://arxiv.org/pdf/1810.04805.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import os.path as opt
import shutil

from typing import List, Text, Any, Optional, Dict, Tuple

from ner_yjcloud.classifiers import INTENT_RANKING_LENGTH
from ner_yjcloud.components import Component
from ner_yjcloud.models.gpu_utils import getGPUs, getAvailabilityGPU
from ner_yjcloud.models import report_epoch_score_matr
from ner_yjcloud.models.bert.bert_finetune import *


try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    from transformers import BertConfig as BC

except ImportError:
    raise ImportError("No torch package found")

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

from ner_yjcloud.utils import write_to_file
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data import TrainingData
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message

BERTPATH = "bert"

class BertIntentClassifier(Component):
    """
    Supervised Bert fine-tune model training
    """

    name = "bert_intent_classifier"

    provides = ["intent", "intent_ranking"]

    defaults = {
        # global training params
        "batch_size": 32,
        "max_seq_len": 64,
        "epochs": 30,
        "walking_epoch_visual": 1,
        "lr": 2e-5,
        "dropout": 0.2,
        "folds": 0.8,

        # pre-trained model
        "pretrained": "/home/user/yuanyh/Bert/torch_model",
        "bert_config": "bert_config.json",
        "bert_model": "bert_model.bin",
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
        super().__init__(component_config)
        self.device = self._load_device()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model
        if self.model:
            self.model.to(self.device)

        self.eval_res = {}

        self.batch_size = self.component_config.get("batch_size", 16)
        self.max_seq_len = self.component_config.get("max_seq_len", 64)
        self.epochs = self.component_config.get("epochs", 10)
        self.lr = self.component_config.get("lr", 2e-5)
        self.dropout = self.component_config.get("dropout", 0.2)
        self.folds = min(self.component_config.get("folds", 1.), 1.)

        self.walking_epoch_visual = self.component_config.get("walking_epoch_visual", 1)

        self.pre_path = self.component_config.get("pretrained", "./BerttModel")
        self.bert_config = self.component_config.get("bert_config", "bert_config.json")
        self.bert_model = self.component_config.get("bert_model", "bert_model.bin")
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

        return device


    def _check_encode_status(self):
        status = True
        if not opt.exists(self.pre_path):
            logger.error(f"Not find bert pretrained path {self.pre_path}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_config)):
            logger.error(f"Not find bert config file {self.bert_config}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.bert_model)):
            logger.error(f"Not find bert pretrained model {self.bert_model}")
            status = False

        if not opt.exists(opt.join(self.pre_path, self.vocab_config)):
            logger.error(f"Not find bert pretrained model {self.bert_model}")
            status = False

        if not status:
            raise FileNotFoundError("Not find a legal bert model path")


    def _create_intent_dict(self, training_data: TrainingData) -> Any:
        """pass"""

        # intents = set([x.get("intent") for x in training_data.intent_examples])
        intents = [x.get("intent") for x in training_data.intent_examples]

        int2idx, idx2int = IntentDataset(intents)()
        self.int2idx = int2idx
        self.idx2int = idx2int


    def process(self, message: Message, **kwargs) -> None:
        intent = {"name": None, "confidence": 0.}
        intent_ranking = []

        tokenizer = load_pretrained_tokenizer(self.pre_path)
        decode_pipeline = TrainingPipeLine(device = self.device,
                                           int2idx = self.int2idx,
                                           idx2int = self.idx2int)

        if message.text.strip():
            score, label = decode_pipeline.decode(model = self.model,
                                                  tokenizer = tokenizer,
                                                  max_len = self.max_seq_len,
                                                  text = message.text,
                                                  ranks = INTENT_RANKING_LENGTH)

            intent = {"name": label[0], "confidence": score[0]}

            intent_ranking = [{"name": x, "confidence": y} for x,y in zip(label[1:],
                                                                          score[1:])]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def train(self,
              training_data: TrainingData,
              config: Optional[RasaNLUModelConfig], **kwargs: Any) -> None:

        encoder, tokenizer = load_pretrained(mpath = self.pre_path,
                                             config = self.bert_config,
                                             model = self.bert_model)

        self._create_intent_dict(training_data)

        # if self.folds == 1.:
        #     train_examples, test_examples = training_data, TrainingData()
        # else:
        train_examples, test_examples = training_data.train_test_split(train_frac = self.folds)

        data_loader = NluClsDataLoader(message = train_examples.training_examples,
                                       tokenizer = tokenizer,
                                       max_len = self.max_seq_len,
                                       batch_size = self.batch_size,
                                       label_dict = self.int2idx)

        if test_examples.training_examples:

            test_data_loader = NluClsDataLoader(message = test_examples.training_examples,
                                                tokenizer = tokenizer,
                                                max_len = self.max_seq_len,
                                                batch_size = self.batch_size,
                                                label_dict = self.int2idx)
        else:
            test_data_loader = None

        train_pipeline = TrainingPipeLine(epochs = self.epochs,
                                          walking_epoch_visual = self.walking_epoch_visual,
                                          lr = self.lr,
                                          dropout = self.dropout,
                                          device = self.device,
                                          int2idx = self.int2idx,
                                          idx2int = self.idx2int)


        self.model = train_pipeline.train(encoder,
                                          data_loader = data_loader,
                                          test_loader = test_data_loader)
        self.eval_res = train_pipeline.eval_res


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


            with open(e_path, "rb") as f:
                intent_dict = pickle.load(f)


            # assign weight to fine-tune model
            encoder = load_encoder(t_path, meta.get("bert_config"))

            _state_dict = torch.load(opt.join(t_path, meta.get("bert_model")), map_location = cls._load_device())

            fine_tune_model = BertFineTuneModel(encoder,
                                                len(intent_dict.get("int2idx")))

            fine_tune_model.load_state_dict(state_dict = _state_dict)

            fine_tune_model.eval()

            meta["pretrained"] = t_path

            return BertIntentClassifier(component_config=meta,
                                        model=fine_tune_model,
                                        int2idx=intent_dict.get("int2idx"),
                                        idx2int=intent_dict.get("idx2int"))


    @staticmethod
    def copy_file_to_dir(input_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        shutil.copy2(input_file, output_dir)


    def persist(self, model_dir):
        if self.model is None:
            return {"pretrained": None}

        t_path = opt.join(model_dir, BERTPATH, "bert_model.bin")

        try:
            os.makedirs(opt.dirname(t_path))
        except FileExistsError:
            pass

        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
        torch.save(model_to_save.state_dict(), t_path)


        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.bert_config),
                              output_dir = opt.join(model_dir, BERTPATH))
        self.copy_file_to_dir(input_file = opt.join(self.pre_path,
                                                    self.vocab_config),
                              output_dir = opt.join(model_dir, BERTPATH))

        e_path = opt.join(model_dir, BERTPATH, "intent_dict.pkl")
        with open(e_path, "wb") as f:
            pickle.dump({"int2idx": self.int2idx,
                         "idx2int": self.idx2int},
                        f)

        write_to_file(opt.join(model_dir, BERTPATH, f"eval_{self.name}.txt"),
                      report_epoch_score_matr(self.eval_res))

        return {"pretrained": BERTPATH,
                "bert_model": "bert_model.bin",
                "bert_config": "bert_config.json",
                "vocab_config": "vocab.txt",
                "intent_dict": "intent_dict.pkl"}
