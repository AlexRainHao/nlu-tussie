'''
Usage: NER based on token POS extracted from `Jieba`

This method not recommended because of much wrong of pos label exists in default jieba `dict.txt`

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings

from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_tussie import utils
from ner_tussie.extractors import EntityExtractor
from ner_tussie.model import Metadata
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData
from ner_tussie.utils import write_json_to_file



class JiebaPsegExtractor(EntityExtractor):
    name = "jieba_pseg_extractor"

    provides = ["entities"]

    defaults = {
        "part_of_speech": ['nr'] # nr：人名，ns：地名，nt：机构名
    }

    def __init__(self, component_config=None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(JiebaPsegExtractor, self).__init__(component_config)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.posseg_cut_examples(message))
        
        message.set("entities", extracted, add_to_output=True)


    def posseg_cut_examples(self, example):
        raw_entities = example.get("entities", [])
        example_posseg = self.posseg(example.text)

        for (item_posseg, start, end) in example_posseg:
            part_of_speech = self.component_config["part_of_speech"]
            for (word_posseg, flag_posseg) in item_posseg:
                if flag_posseg in part_of_speech:
                    raw_entities.append({
                        'start': start,
                        'end': end,
                        'value': word_posseg,
                        'entity': flag_posseg
                    })
        return raw_entities

    @staticmethod
    def posseg(text):
        # type: (Text) -> List[Token]

        USER_DICT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dict.txt')

        import jieba
        import jieba.posseg as pseg
        jieba.load_userdict(USER_DICT)

        result = []
        for (word, start, end) in jieba.tokenize(text):
            pseg_data = [(w, f) for (w, f) in pseg.cut(word)]
            result.append((pseg_data, start, end))

        return result

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)
