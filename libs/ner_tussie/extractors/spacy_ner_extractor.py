'''
Usage: NER based on pre-trained `Spacy` model

References as:
    https://spacy.io

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, Dict, Optional, Text

import os
import re
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from configparser import ConfigParser


from ner_tussie.extractors import EntityExtractor, EntityObj, ConjuncEntityDynamic
from ner_tussie.model import Metadata
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData


class preSpacyEntityConfig:

    interest_entities = {
        "PER": ["PERSON", "PER"],
        "RESIDENT": ["FAC", "GPE", "LOC"],
        "ORG": ["ORG"],
        "Money": ["MONEY"],
        "Date": ["DATE", "TIME"],
        "TITLE": ["TITLE"],
        "EDU": ["EDU"]
    }

    conjunction_flag = ['SPACE']

    patterns = {"RESIDENT": [["[门幢楼栋室巷屯乡镇元层区庄址村]$", "conj"]],
                "PER": [["[\da-zA-Z]", "clear"]],
                "Money": [["[^0-9点\.,，多几十百千万亿余一二两三四五六七八九零]+元$", "clear"]],
                "Date": [["[0-9一二三四五六七八九十零]+岁", "clear"],
                         ["百分", "clear"],
                         ["[万亿例]$", "clear"]]}

    configComp = {"interest_entities": interest_entities,
                  "confidence_threshold": 0.7,
                  "conjunction_flag": conjunction_flag,
                  "patterns": patterns}
    


class SpacyEntityExtractor(EntityExtractor):
    name = "spacy_entity_extractor"

    provides = ["entities"]

    requires = ["tokens", "pos", "preEntity"]

    defaults = preSpacyEntityConfig.configComp

    def __init__(self, component_config = None):

        super(SpacyEntityExtractor, self).__init__(component_config)
        self.component_config = component_config

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.spacyExtract(message))

        message.set("entities", extracted, add_to_output = True)


    @classmethod
    def load(cls,
             model_dir = None, # type: Optional[Text]
             model_metadata = None, # type: Optional[Metadata]
             cached_component = None, # type: Optional[Component]
             **kwargs # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)

    @staticmethod
    def _reverse_config(config_dict):
        reverse_dict = {}
        for key, value in config_dict.items():
            for _val in value:
                reverse_dict.setdefault(_val, key)

        return reverse_dict


    def spacyExtract(self, example):
        raw_entity = example.get('entities', [])
        text = example.text
        tokens = example.get('tokens')
        pos = example.get('pos').get('pos')
        pre_entities = example.get('preEntity')

        mapping_label_dict = self.component_config.get("interest_entities")
        conj_config = self.component_config.get("conjunction_flag")
        confidence_threshold = self.component_config.get('confidence_threshold')
        patterns = self.component_config.get("patterns")

        if text.strip() == '':
            return raw_entity

        if not pre_entities:
            return raw_entity

        # define post_processing function
        postProgress = ConjuncEntityDynamic(tokens, pre_entities, pos, 
                                            threShold = confidence_threshold,
                                            initialConf = 0.)

        # register for label mapping
        for key, value in mapping_label_dict.items():
            postProgress.register(0, src = value, tar = key)

        # register for conjunction
        postProgress.register(1, src = conj_config)

        # register for regex patterns
        for key, value in patterns.items():
            postProgress.register(2, src = key, tar = value)

        raw_entity = postProgress.main(raw_entity = raw_entity)

        return raw_entity
