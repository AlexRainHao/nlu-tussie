from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import os
import re
from uuid import uuid4
import json
from typing import Any, Dict, List, Optional, Text

import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from nlu_tussie import utils
from nlu_tussie.featurizers import Featurizer
from nlu_tussie.training_data import Message
from nlu_tussie.components import Component
from nlu_tussie.model import Metadata
from bert_serving.client import BertClient
import requests

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BertVectorsFeaturizer(Featurizer):
    name = "bert_vectors_featurizer"

    provides = ["text_features"]

    defaults = {
        "ip": 'localhost',
        "port": 5555,
        "port_out": 5556,
        "http_port": 5557,
        "show_server_config": False,
        "output_fmt": 'ndarray',
        "check_version": True,
        "timeout": 5000,
        "identity": None,
        "batch_size": 128
    }

    @classmethod
    def required_packages(cls):
        return ["numpy", "bert_serving"]

    def __init__(self, component_config=None):
        super(BertVectorsFeaturizer, self).__init__(component_config)
        ip = self.component_config['ip']
        port = self.component_config['port']
        port_out = self.component_config['port_out']
        http_port = self.component_config['http_port']
        show_server_config = self.component_config['show_server_config']
        output_fmt = self.component_config['output_fmt']
        check_version = self.component_config['check_version']
        timeout = self.component_config['timeout']
        identity = self.component_config['identity']
        self.bc = BertClient(
            ip=ip,
            port=int(port),
            port_out=int(port_out),
            show_server_config=show_server_config,
            output_fmt=output_fmt,
            check_version=check_version,
            timeout=int(timeout),
            identity=identity
        )

    @staticmethod
    def _gene_id():
        return str(uuid4()).replace('-', '')

    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        return BertVectorsFeaturizer(component_conf)

    def _get_message_text(self, message):
        all_tokens = []

        for msg in message:
            all_tokens.append(msg.text)

        # bert_embedding = self.bc.encode(all_tokens, is_tokenized=False)

        postOp = requests.request("POST", url='http://%s:%s/encode' % (self.component_config['ip'],
                                                                           self.component_config['http_port']),
                                  headers={'Content-Type': 'application/json'}, data = json.dumps({
                'id': self._gene_id(),
                'texts': [[x] for x in all_tokens],
                'is_tokenized': "false"
            }))

        return np.squeeze(json.loads(postOp.text)['result'])
        # return np.squeeze(bert_embedding)

    def train(self, training_data, cfg=None, **kwargs):
        batch_size = self.component_config['batch_size']

        epochs = len(training_data.intent_examples) // batch_size + \
            int(len(training_data.intent_examples) % batch_size > 0)

        for ep in tqdm(range(epochs), desc="Epochs"):
            end_idx = (ep + 1) * batch_size
            start_idx = ep * batch_size
            examples = training_data.intent_examples[start_idx:end_idx]
            tokens_text = self._get_message_text(examples)
            X = np.array(tokens_text)

            for i, example in enumerate(examples):
                example.set(
                    "text_features", self._combine_with_existing_text_features(example, X[i]))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message_text = self._get_message_text([message])

        message.set("text_features", self._combine_with_existing_text_features(
            message, message_text))

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)
