'''
Usage: extra Baidu NLP package for supporting ner

The tokenizer and NER is non-dependent on the others pipeline
because we regard as important it's LOC ner ability

=======
References as:
    https://github.com/baidu/lac

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, Dict, Optional, Text

import os
import os.path as opt
import sys
import re
from LAC import LAC

# LIBPATH = opt.abspath(opt.join(opt.abspath(__file__), *[opt.pardir] * 2))
# sys.path.append(LIBPATH)


from nlu_tussie.extractors import EntityExtractor, EntityObj, ConjuncEntityDynamic
from nlu_tussie.model import Metadata
from nlu_tussie.training_data import Message
from nlu_tussie.training_data import TrainingData


class preLacEntityConfig:

    mapping_label_dict = {"PER": ["PER"],
                          "RESIDENT": ["LOC"],
                          "ORG": ["ORG"],
                          "Date": ["TIME"]}

    conjunction_flag = ['w', 'SPACE']

    patterns = {"RESIDENT": [["[门幢楼栋室巷屯乡镇元层区庄址村]$", "conj"]],
                "PER": [["[\da-zA-Z]", "clear"]],
                "ORG": [["[京津冀晋蒙辽吉黑沪苏浙皖闽赣甘青宁新鲁豫鄂湘粤桂琼渝川黔滇藏陕台港澳]a-zA-Z0-9+", "clear"]]}

    configComp = {"interest_entities": mapping_label_dict,
                  "confidence_threshold": 0.7,
                  "conjunction_flag": conjunction_flag,
                  "patterns": patterns}



class LacEntityExtractor(EntityExtractor):
    name = "lac_entity_extractor"

    provides = ["entities"]

    defaults = preLacEntityConfig.configComp

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(LacEntityExtractor, self).__init__(component_config)
        self.component_config = component_config

        self.lac = LAC(mode = 'lac')


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.lacExtract(message))

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
    def add_unique_entity(target_list):
        # if entity not in target_list:
        #     target_list.append(entity)
        res_target_list = []
        for target in target_list:
            if target not in res_target_list:
                res_target_list.append(target)

        return res_target_list


    def lacExtract(self, example):
        raw_entity = example.get("entities", [])
        text = example.text

        if text.strip() == '':
            return raw_entity

        mapping_label_dict = self.component_config.get('interest_entities')
        conj_config = self.component_config.get('conjunction_flag')
        confidence_threshold = self.component_config.get('confidence_threshold')
        patterns = self.component_config.get('patterns')

        lac_result_tokens, lac_result_pos = self.lac.run(text)

        #if not list(filter(lambda x: x in mapping_label_dict.keys(), lac_result_pos)):
        #    return raw_entity

        # define post_processing function
        postProgress = ConjuncEntityDynamic(lac_result_tokens, lac_result_pos,
                                            threShold = 0.,
                                            initialConf = 1.)
        postProgress.convert_token()

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
        
