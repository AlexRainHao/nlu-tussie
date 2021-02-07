# -*- coding: utf-8 -*-
'''
Usage: Regular pattern, window search from model parsed results NER for

    * Money

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re

from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_tussie.extractors import EntityExtractor
from ner_tussie.model import Metadata
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData


class preRegexMoney:
    worth_candi = "([值|计|为]+(人民币)?[0-9点\.,，多几十百千万亿余一二两三四五六七八九零]+[欧美日]?[元块]+[左右]{0,2})"
    count_candi = "((花了)?(人民币)?[0-9点\.,，多几十百千万亿余一二两三四五六七八九零]+[欧美日]?[元块]+[左右]{0,2})"
    income_candi = "收入([0-9点\.,，多几十百千万亿余一二两三四五六七八九零]+[欧美日]?[元块]?)"

    configComp = {"worth_dsr_candi": worth_candi,
                  "count_dsr_candi": count_candi,
                  "income_dsr_candi": income_candi,
                  "tagName": "Money"}


class MoneyExtractor(EntityExtractor):
    name = "money_extractor"

    provides = ["entities"]

    requires = ['pos']

    defaults = preRegexMoney.configComp

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(MoneyExtractor, self).__init__(component_config)
        self.component_config = component_config
        # self.tagName = "Money"


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.moneyExtract(message))

        message.set("entities", extracted, add_to_output = True)


    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)


    @staticmethod
    def _handle_config(config_dict):
        config_dict["worth_dsr_candi"] = re.compile(config_dict["worth_dsr_candi"])
        config_dict["count_dsr_candi"] = re.compile(config_dict["count_dsr_candi"])
        config_dict["income_dsr_candi"] = re.compile(config_dict["income_dsr_candi"])

        return config_dict

    def moneyExtract(self, example):

        raw_entity = example.get('entities', [])
        text = example.text

        if len(text) == 1:
            return raw_entity

        pointer = 0
        length = len(text)

        while pointer < length - 1:
            _str = text[pointer:]

            _search_1_flag = re.compile(self.component_config['worth_dsr_candi']).search(_str)
            _search_2_flag = re.compile(self.component_config['count_dsr_candi']).search(_str)
            _search_3_flag = re.compile(self.component_config['income_dsr_candi']).search(_str)

            if _search_1_flag \
                    and _search_2_flag \
                    and _search_1_flag.span()[0] > _search_2_flag.span()[0]:
                _search_1_flag, _search_2_flag = _search_2_flag, _search_1_flag


            if _search_1_flag:
                span = _search_1_flag.span()

                _value = _search_1_flag.group(1)
                pos_adder = 0
                if _value[0] in ["值", "计", "为"]:
                    pos_adder = 1
                    _value = _value[1:]

                elif _value[0:2] == "花了":
                    pos_adder = 2
                    _value = _value[2:]

                if _value != "多元":
                    raw_entity.append({"start": pointer + span[0] + pos_adder,
                                        "end": pointer + span[1],
                                        "value": _value,
                                        "entity": self.component_config.get('tagName', 'Money'),
                                       "confidence": 1.})
                pointer += span[1]
                continue

            elif _search_2_flag:
                span = _search_2_flag.span()

                _value = _search_2_flag.group(1)
                pos_adder = 0
                if _value[0] in ["值", "计", "为"]:
                    pos_adder = 1
                    _value = _value[1:]

                elif _value[0:2] == "花了":
                    pos_adder = 2
                    _value = _value[2:]

                if _value != "多元":
                    raw_entity.append({"start": pointer + span[0] + pos_adder,
                                       "end": pointer + span[1],
                                       "value": _value,
                                       "entity": self.component_config.get('tagName', 'Money'),
                                       "confidence": 1.})
                pointer += span[1]
                continue

            elif _search_3_flag:
                span = _search_3_flag.span()
                _value = _search_3_flag.group(1)
                raw_entity.append({"start": pointer + span[0],
                                   "end": pointer + span[1],
                                   "value": _value,
                                   "entity": self.component_config.get('tagName', 'Money'),
                                   "confidence": 1.})
                pointer += span[1]
                continue

            else:
                return raw_entity

        return raw_entity

