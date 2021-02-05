from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
import os
import glob
import codecs

from ner_yjcloud.components import Component
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data import Message
from ner_yjcloud.training_data import TrainingData

logger = logging.getLogger(__name__)

SPACY_CUSTOM_DICTIONARY_PATH = "tokenizer_spacy"
SPACY_CUSTOM_MODEL_PATH = "CoreModel"

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from ner_yjcloud.model import Metadata


class SpacyNLP(Component):
    name = "nlp_spacy"

    provides = ["spacy_doc", "spacy_nlp"]

    language_list = ["zh", "en"]

    defaults = {
        # name of the language model to load - if it is not set
        # we will be looking for a language model that is named
        # after the language of the model, e.g. `en`
        "model": "/home/user/yuanyh/vitual/test_ner/model/CoreModel",

        # when retrieving word vectors, this will decide if the casing
        # of the word is relevant. E.g. `hello` and `Hello` will
        # retrieve the same vector, if set to `False`. For some
        # applications and models it makes sense to differentiate
        # between these two words, therefore setting this to `True`.
        "case_sensitive": False,

        "user_dictionary_path": None
    }

    def __init__(self, component_config=None, nlp=None):
        # type: (Dict[Text, Any], Language) -> None

        self.nlp = nlp
        super(SpacyNLP, self).__init__(component_config)

        self.user_dictionary_path = self.component_config.get('user_dictionary_path')
        self.user_model = self.component_config.get('model')


    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy"]

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> SpacyNLP
        import spacy
        from spacy.pipeline import EntityRuler

        component_conf = cfg.for_component(cls.name, cls.defaults)
        spacy_model_name = component_conf.get("model")

        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            spacy_model_name = cfg.language
            component_conf["model"] = cfg.language

        logger.info("Trying to load spacy model with "
                    "name '{}'".format(spacy_model_name))

        nlp = spacy.load(spacy_model_name, parser=False)
        ruler = EntityRuler(nlp, overwrite_ents = True)

        # ==============
        # add self defined word
        nlp = cls.add_self_tokens(nlp, ruler, component_conf.get("user_dictionary_path"))

        cls.ensure_proper_language_model(nlp)
        return SpacyNLP(component_conf, nlp)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        component_meta = model_metadata.for_component(cls.name)

        # Fallback, use the language name, e.g. "en",
        # as the model name if no explicit name is defined
        spacy_model_name = component_meta.get("model", model_metadata.language)

        return spacy_model_name

    @staticmethod
    def add_self_tokens(nlp, ruler, user_dict_path):
        """
        Load the custom dictionaries stored in the path
        each of file have a rows formats as
        yyh PERSON\n
        """
        spacy_userdicts = glob.glob("{}/*".format(user_dict_path))
        user_dict_lines = []
        for _userdict in spacy_userdicts:
            logger.info("Loading Spacy user dictionary at {}".format(_userdict))
            try:
                with codecs.open(_userdict, encoding = 'utf-8') as f:
                    user_dict_lines += f.read().splitlines()
            except:
                logger.warning("Loading Spacy user dictionary {} Failed,"
                               "Plz check it coding of utf-8 and others".format(_userdict))

        new_tokens = []
        new_patterns = []
        for row in user_dict_lines:
            _tokens = row.split(' ')
            new_tokens.append(' '.join(_tokens[:-1]))
            new_patterns.append({"label": _tokens[-1],
                                 "pattern": ' '.join(_tokens[:-1])})

        # 1. add new token for tokenizer
        # Only useful for tokenizer under zh that has pkuseg tokenizer
        try:
            logger.info("Update [%d] self-defined tokens to model tokenizer" % len(new_tokens))
            nlp.tokenizer.pkuseg_update_user_dict(words=new_tokens)
        except:
            logger.warning("Update self-defined tokens Failed")

        # 2. add new NER pattern for tokenizer
        try:
            logger.info("Update [%d] self-defined NER patterns to model" % len(new_patterns))
            ruler.add_patterns(new_patterns)
            nlp.add_pipe(ruler)
        except:
            logger.warning("Update self-defined NER patterns Failed")

        return nlp


    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"spacy_nlp": self.nlp}

    def doc_for_text(self, text):
        """parse each sentence from spacy model"""
        if self.component_config.get("case_sensitive"):
            return self.nlp(text)
        else:
            return self.nlp(text.lower())

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))
            example.set("spacy_nlp", self.nlp)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("spacy_doc", self.doc_for_text(message.text))
        message.set("spacy_nlp", self.nlp)

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        # type: (Text, Metadata, Optional[SpacyNLP], **Any) -> SpacyNLP

        # if cached_component:
        #     return cached_component

        import spacy
        from spacy.pipeline import EntityRuler

        component_meta = model_metadata.for_component(cls.name)
        model_name = component_meta.get("model")
        user_dict_path = component_meta.get("user_dictionary_path")

        if user_dict_path is not None:
            user_dict_path = os.path.join(model_dir, user_dict_path)

            component_meta["user_dictionary_path"] = user_dict_path
        #

        if model_name == SPACY_CUSTOM_MODEL_PATH:
            # model_name = os.path.abspath(os.path.join(__file__, (os.path.pardir + '/') * 4, 'model/CoreModel'))
            model_name = os.path.join(model_dir, model_name)

            component_meta["model"] = model_name

        nlp = spacy.load(model_name, parser=False)
        ruler = EntityRuler(nlp, overwrite_ents=True)


        nlp = cls.add_self_tokens(nlp, ruler, user_dict_path)
        cls.ensure_proper_language_model(nlp)
        return cls(component_meta, nlp)

    @staticmethod
    def ensure_proper_language_model(nlp):
        # type: (Optional[Language]) -> None
        """Checks if the spacy language model is properly loaded.
        Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception("Failed to load spacy language model. "
                            "Loading the model returned 'None'.")
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception("Failed to load spacy language model for "
                            "lang '{}'. Make sure you have downloaded the "
                            "correct model (https://spacy.io/docs/usage/)."
                            "".format(nlp.lang))


    @staticmethod
    def copy_files_dir_to_dir(input_dir, output_dir):
        import shutil
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob("{}/*".format(input_dir))
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    @staticmethod
    def copy_dir_to_dir(input_dir, output_dir):
        from distutils.dir_util import copy_tree

        copy_tree(input_dir, output_dir)


    def persist(self, model_dir):

        src_model_path = self.user_model
        relative_model_path = src_model_path

        if self.user_dictionary_path is not None:
            """copy user dictionary"""
            target_dictionary_path = os.path.join(model_dir,
                                                  SPACY_CUSTOM_DICTIONARY_PATH)

            self.copy_files_dir_to_dir(self.user_dictionary_path,
                                       target_dictionary_path)

        if os.path.exists(src_model_path):
            """copy core model"""
            target_model_path = os.path.join(model_dir, SPACY_CUSTOM_MODEL_PATH)

            self.copy_dir_to_dir(self.user_model, target_model_path)

            relative_model_path = SPACY_CUSTOM_MODEL_PATH

        return {"user_dictionary_path": SPACY_CUSTOM_DICTIONARY_PATH,
                "model": relative_model_path}
