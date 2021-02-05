"""
Usage: Intent Classifier through certain KeyWords

The keyword represented by regular patterns, that should written into training data file

E.X.
    "regex_intent": [
      {
        "intent": "greet",
        "regex": ["您好", "你好"]
      }
    ]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Text, Any, Optional, Dict, Tuple
import re
import os.path as opt
import codecs
import json
import logging

from ner_yjcloud.classifiers import INTENT_RANKING_LENGTH
from ner_yjcloud.components import Component
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data import TrainingData, Message
from ner_yjcloud.model import Metadata

logger = logging.getLogger(__name__)

class KeyWordIntentClassifier(Component):
    """
    Intent Classifier through certain KeyWords,
    a parameter named `regex_intent` would load during training or inference procedure

    condition 1.
        The Training, `regex_intent` could be empty string and all the keyword patterns loaded from training data file

    condition 2.
        The Inference, `regex_intent` loaded from model metadata configuration file, which a file name originally,
        and converted to pair of keyword and intent as dictionary format
    """

    name = "keyword_intent_classifier"

    provides = ["intent"]

    defaults = {"regex_intent": ""}

    def __init__(self, component_config):
        super(KeyWordIntentClassifier, self).__init__(component_config)

        self.regex_intent = self.load_intent_file(self.component_config.get("regex_intent"))

    @staticmethod
    def load_intent_file(filename):
        if filename.strip() == "":
            return {}

        elif not opt.exists(filename):
            return {}

        else:
            return json.load(codecs.open(filename, encoding = 'utf-8'))

    def train(self,
              training_data: TrainingData,
              config: Optional[RasaNLUModelConfig],
              **kwargs: Any):

        self.regex_intent = training_data.regex_intent
        logger.info(f"keyword intent classifier extracted {len(self.regex_intent)} regex patterns and trained done")


    def process(self, message: Message, **kwargs: Any) -> None:
        _intent = {"name": None, "confidence": 0.}

        if message.text.strip():
            intent = self.parser(message.text)

            if intent:
                message.set("intent", intent, add_to_output = True)

            elif message.get("intent"):
                pass

            else:
                message.set("intent", _intent, add_to_output = True)

        else:
            message.set("intent", _intent, add_to_output = True)
            message.set("intent_ranking", [], add_to_output = True)


    def parser(self, msg: Text):
        intent = {"name": None, "confidence": 0.}

        for _rgx, _int in self.regex_intent.items():
            if re.search(_rgx, msg):
                intent["name"] = _int
                intent["confidence"] = 1.
                return intent

        return False

    @classmethod
    def load(cls,
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional[Component] = None,
             **kwargs: Any) -> Any:

        meta = model_metadata.for_component(cls.name)

        if model_dir:

            meta["regex_intent"] = opt.join(model_dir, meta.get("regex_intent"))

            return KeyWordIntentClassifier(component_config = meta)

        else:

            logger.error(f"Failed load regex intent dictionary file {meta.get('regex_intent')}")
            meta["regex_intent"] = ""
            return KeyWordIntentClassifier(component_config = meta)

    def persist(self, model_dir: Text):
        filename = opt.join(model_dir, "regex_intent.json")

        with codecs.open(filename, 'w', encoding = 'utf-8') as f:
            json.dump(self.regex_intent, f, ensure_ascii = False, indent = 2)

        logger.info(f"regex intent dictionary persisted to {filename}")

        return {"regex_intent": "regex_intent.json"}