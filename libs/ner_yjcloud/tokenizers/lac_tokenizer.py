'''
Usage:lac tokenizer for fast training and deployment

the tokenizer is better than either `jieba` or `spacy`
and dependency of lac is smaller, which lead us to a
lightweight and directive deployment
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any, List, Tuple
import logging
from LAC import LAC
import os
import sys

# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_yjcloud.components import Component
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.tokenizers import Tokenizer, Token, SegPos, PreEntity
from ner_yjcloud.training_data import Message
from ner_yjcloud.training_data import TrainingData

logger = logging.getLogger(__name__)

SPACE_FLAG = "SPACE"

class LacTokenizer(Tokenizer, Component):
    name = "tokenizer_lac"

    provides = ["tokens", "pos", "preEntity"]

    language_list = ["zh", "en"]

    def __init__(self, component_config = None):
        # type: (Dict[Text, Any]) -> None

        super(LacTokenizer, self).__init__(component_config)

        self.lac = LAC(mode = 'lac')

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["LAC"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            token_res, pos_res, entity_res = self.tokenize(example.text)

            example.set("tokens", token_res)
            example.set("pos", pos_res)
            example.set("preEntity", entity_res)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        token_res, pos_res, entity_res = self.tokenize(message.text)

        message.set('tokens', token_res)
        message.set('pos', pos_res)
        message.set('preEntity', entity_res)


    def tokenize(self, text):
        tokens, pos = self.lac.run(text)

        tokenCls, entityCls = [], []

        start_pt = 0

        for _tok, _pos in zip(tokens, pos):
            tokenCls.append(Token(_tok, start_pt))
            entityCls.append(PreEntity(start_id = start_pt,
                                       end_id = start_pt + len(_tok),
                                       text = _tok,
                                       entity = SPACE_FLAG if _pos == 'w' else _pos,
                                       confidence = 1.))

            start_pt += len(_tok)

        return tokenCls, SegPos(zip(tokens, pos)), entityCls

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> LacTokenizer

        meta = model_metadata.for_component(cls.name)


        return cls(meta)
