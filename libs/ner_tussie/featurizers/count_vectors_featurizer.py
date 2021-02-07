"""
Usage: BOW Sparse Representation features

Based on `sklearn` from:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

supports:
    * raw text normalizer, defaults convert numbers to '__NUMBER__'
    * self-defined tokenizer if needed, load from used package name
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import os
import os.path as opt
import re
from typing import Any, Dict, List, Optional, Text, Callable, Union

import sys
import importlib
import codecs
import pickle

from ner_tussie import utils
from ner_tussie.featurizers import Featurizer
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData
from ner_tussie.components import Component
from ner_tussie.config import RasaNLUModelConfig
from ner_tussie.model import Metadata

from ner_tussie.normalizer import Normalizers

try:
    from sklearn.feature_extraction.text import CountVectorizer

except:
    raise ImportError("No sklearn package found")

logger = logging.getLogger(__name__)


class CountVectorsFeaturizer(Featurizer):
    """pass"""

    name = "cout_vectors_featurizer"

    provides = ["text_features"]

    defaults = {
        # args default as sklearn CountVectorizer
        "token_pattern": r'(?u)\b\w\w+\b',
        "tokenizer": None, # self-defined tokenizer, e.x. "jieba.lcut"
        "normalizer": {"digit_zero_shrink": True}, # series normalizers

        "strip_accents": None,

        "stop_words": None, # file path or tokens list or keyword="english"

        "min_df": 2,
        "max_df": 1.0,

        "min_ngram": 1,
        "max_ngram": 1,
        "max_features": None,

        "lowercase": True,

        "OOV_token": None,
        "OOV_words": []
    }

    @staticmethod
    def _read_file_content(fpath):
        """check whether file exists, and then load content"""
        lines = []
        try:
            with codecs.open(fpath, encoding = "utf-8") as f:
                lines = f.read().strip().split()

        except:
            logger.warning("load stop word dictionary failed,"
                           "and self defined stop word would ignored")

        return lines

    @classmethod
    def required_packages(cls):  # type: () -> List[Text]
        return ["sklearn"]

    def _load_global_params(self):
        for key, val in self.component_config.items():
            self.__setattr__(key, val)

    def _load_OOV_params(self):
        self.OOV_token = self.component_config.get("OOV_token", None)
        self.OOV_words = self.component_config.get("OOV_words", [])

        if self.OOV_words and not self.OOV_token:
            logger.error("The list OOV_words={} was given, but "
                         "OOV_token was not. OOV words are ignored."
                         "".format(self.OOV_words))
            self.OOV_words = []

        if self.lowercase and self.OOV_token:
            # convert to lowercase
            self.OOV_token = self.OOV_token.lower()
            if self.OOV_words:
                self.OOV_words = [w.lower() for w in self.OOV_words]

    def _load_tokenizer(self) -> Union[Callable, None]:
        """load self-defined tokenizer if assigned"""
        string = self.component_config.get("tokenizer", None)

        if not string:
            self.tokenizer_op = None

        else:
            par_pkg = string.split('.')

            op = importlib.import_module(par_pkg.pop(0))

            while par_pkg:
                op = op.__dict__[par_pkg.pop(0)]

            self.tokenizer_op = op

    def _load_stop_words(self) -> Union[None, List[Text]]:
        """load stop words from config"""

        if self.stop_words == "english":
            return

        elif isinstance(self.stop_words, str):
            # for file path
            self.stop_words = self._read_file_content(self.stop_words)

        elif isinstance(self.stop_words, list):
            # for token list
            return

        else:
            logger.warning("stop word setting illegal so would be ignored")
            self.stop_words = []


    def __init__(self,
                 component_config = None,
                 vec = None):
        super().__init__(component_config)

        self._load_global_params()
        self._load_OOV_params()
        self._load_tokenizer()
        self._load_stop_words()

        self.normalizer = Normalizers().parse_dict_config(self.component_config.get("normalizer", {}))

        self.vec = vec

    def _tokenizer(self, text):
        """Override tokenizer in CountVectorizer

        Parameters
        ----------
        text : Text, E.X. "today is a good day"
        """

        # if normalizer
        for func in self.normalizer:
            text = func(text)

        # if self defined tokenizer
        if self.tokenizer_op:
            tokens = self.tokenizer_op(text.replace(' ', ''))

        else:
            tokens = re.compile(self.token_pattern).findall(text)

        if self.OOV_token:
            if hasattr(self.vec, "vocabulary_"):
                # CountVectorizer is trained, process for prediction
                if self.OOV_token in self.vec.vocabulary_:
                    tokens = [
                        t if t in self.vec.vocabulary_.keys()
                        else self.OOV_token for t in tokens
                    ]

            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                tokens = [
                    self.OOV_token if t in self.OOV_words else t for t in tokens
                ]

        return tokens

    def _get_message_text(self, message: Message):
        """tokens should obatined from fore-pipeline expected"""

        if message.get("tokens"):  # if directly tokens is provided
            return ' '.join([t.text for t in message.get("tokens")])
        else:
            return message.text

    def train(self,
              training_data: TrainingData,
              config: Optional[RasaNLUModelConfig], **kwargs: Any) -> None:
        """pass"""

        self.vec = CountVectorizer(token_pattern = self.token_pattern,
                                   strip_accents = self.strip_accents,
                                   lowercase = self.lowercase,
                                   stop_words = self.stop_words,
                                   ngram_range = (self.min_ngram,
                                                  self.max_ngram),
                                   max_df = self.max_df,
                                   min_df = self.min_df,
                                   max_features=self.max_features,
                                   tokenizer = self._tokenizer)

        exams = [self._get_message_text(msg) for msg in training_data.intent_examples]

        X = self.vec.fit_transform(exams).toarray() # [num, vocab_size]

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set("text_features",
                        self._combine_with_existing_text_features(example,
                                                                  X[i]))

    def process(self, message: Message, **kwargs) -> None:
        """prediction"""

        if self.vec is None:
            logger.error("No model loade, and count vectors features pipeline ignored")
            return

        msg = self._get_message_text(message)

        fea = self.vec.transform([msg]).toarray().squeeze()

        message.set("text_features", self._combine_with_existing_text_features(
            message, fea
        ))

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("featurizer_file"):
            # with open(opt.join(model_dir, meta.get("featurizer_file")), "rb") as f:
            #     vec = pickle.load(f)

            # return CountVectorsFeaturizer(meta, vec)

            logger.info("Load Count Vector Featurizer model done")
            featurizer_file = opt.join(model_dir, meta.get("featurizer_file"))
            return utils.pycloud_unpickle(featurizer_file)

        else:
            logger.warning("Failed to load trained count features model")
            return CountVectorsFeaturizer(meta)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        """save model"""

        # save model
        self.tokenizer_op = None

        model_name = opt.join(model_dir, self.name + ".pkl")
        # with open(model_name, "wb") as f:
        #     pickle.dump(self.vec, f)

        utils.pycloud_pickle(model_name, self)

        # save stop word
        if self.stop_words == "english":
            return {
                "featurizer_file": self.name + ".pkl",
            }

        sw_name = opt.join(model_dir, self.name + '_stopword.txt')
        with codecs.open(sw_name, 'wb', encoding='utf-8') as f:
            f.writelines(self.stop_words)

        return {
            "featurizer_file": self.name + ".pkl",
            "stop_words": self.name + '_stopword.txt'
        }

