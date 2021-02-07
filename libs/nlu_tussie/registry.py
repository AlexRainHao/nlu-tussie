"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(LIBPATH)

import typing
from nlu_tussie import utils
from typing import Any
from typing import Optional
from typing import Text
from typing import Type

from nlu_tussie.model import Metadata

# ====================
# tokenizer
from nlu_tussie.utils.spacy_utils import SpacyNLP
from nlu_tussie.tokenizers.spacy_tokenizer import SpacyTokenizer
from nlu_tussie.tokenizers.jieba_tokenizer import JiebaTokenizer
from nlu_tussie.tokenizers.lac_tokenizer import LacTokenizer

# ====================
# features
from nlu_tussie.featurizers.intent_featurizer_wordvector import WordVectorsFeaturizer  # customize
from nlu_tussie.featurizers.bert_vectors_featurizer import BertVectorsFeaturizer  # customize
from nlu_tussie.featurizers.ngram_featurizer import NGramFeaturizer
from nlu_tussie.featurizers.regex_featurizer import RegexFeaturizer
from nlu_tussie.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

# ====================
# extractors
from nlu_tussie.extractors.entity_synonyms import EntitySynonymMapper
from nlu_tussie.extractors.crf_entity_extractor import CRFEntityExtractor
from nlu_tussie.extractors.bilstm_crf_extractor import BiLSTMCrfEntityExtractor # customize
from nlu_tussie.extractors.jieba_pseg_extractor import JiebaPsegExtractor  # customize
from nlu_tussie.extractors.spacy_ner_extractor import SpacyEntityExtractor # customize
from nlu_tussie.extractors.lac_ner_extractor import LacEntityExtractor # customize
from nlu_tussie.extractors.lattice_extractor import LatticeNerEntityExtractor # customize

from nlu_tussie.extractors.nerMoney import MoneyExtractor # customize
from nlu_tussie.extractors.nerNumber import NumberPatternExtractor # customize
from nlu_tussie.extractors.nerIdentity import IdentityPatternExtractor # customize
from nlu_tussie.extractors.nerLawAbout import LawAboutExtractor # customize

# ====================
# classifier
from nlu_tussie.classifiers.albert_classifier import AlbertIntentClassifier
from nlu_tussie.classifiers.bert_classifier import BertIntentClassifier
from nlu_tussie.classifiers.keyword_classifier import KeyWordIntentClassifier
from nlu_tussie.classifiers.svm_classifier import SklearnIntentClassifier
from nlu_tussie.classifiers.embedding_classifier import StarSpaceEmbeddingClassifier

if typing.TYPE_CHECKING:
    from nlu_tussie.components import Component
    from nlu_tussie.config import RasaNLUModelConfig, RasaNLUModelConfig

# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    SpacyNLP,
    SpacyTokenizer,
    JiebaTokenizer,
    LacTokenizer, # customize
    WordVectorsFeaturizer,  # customize
    BertVectorsFeaturizer,  # customize
    NGramFeaturizer, RegexFeaturizer,
    CountVectorsFeaturizer,
    EntitySynonymMapper,
    CRFEntityExtractor,
    BiLSTMCrfEntityExtractor, # customize
    JiebaPsegExtractor, # customize
    SpacyEntityExtractor, # customize
    LacEntityExtractor, # customize
    LatticeNerEntityExtractor, #customize
    MoneyExtractor, # customize
    NumberPatternExtractor, # customize
    IdentityPatternExtractor, # customize
    LawAboutExtractor, # customize
    AlbertIntentClassifier, # customize
    BertIntentClassifier, # customize
    KeyWordIntentClassifier, # customize
    SklearnIntentClassifier, # customize
    StarSpaceEmbeddingClassifier, # customize
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}

# To simplify usage, there are a couple of model templates, that already add
# necessary components in the right order. They also implement
# the preexisting `backends`.
registered_pipeline_templates = {
    "spacy_sklearn": [
        "nlp_spacy",
        "tokenizer_spacy",
        "intent_featurizer_spacy",
        "intent_entity_featurizer_regex",
        "ner_crf",
        "ner_synonyms",
        "intent_classifier_sklearn"
    ],
    "keyword": [
        "intent_classifier_keyword",
    ],
    "tensorflow_embedding": [
        "tokenizer_whitespace",
        "ner_crf",
        "intent_featurizer_count_vectors",
        "intent_classifier_tensorflow_embedding",
        "intent_estimator_classifier_tensorflow_embedding_bert"
    ]
}


def pipeline_template(s):
    components = registered_pipeline_templates.get(s)

    if components:
        # converts the list of components in the configuration
        # format expected (one json object per component)
        return [{"name": c} for c in components]

    else:
        return None


def get_component_class(component_name):
    # type: (Text) -> Optional[Type[Component]]
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        try:
            return utils.class_from_module_path(component_name)
        except Exception:
            raise Exception(
                    "Failed to find component class for '{}'. Unknown "
                    "component name. Check your configured pipeline and make "
                    "sure the mentioned component is not misspelled. If you "
                    "are creating your own component, make sure it is either "
                    "listed as part of the `component_classes` in "
                    "`nlu_tussie.registry.py` or is a proper name of a class "
                    "in a module.".format(component_name))
    return registered_components[component_name]


def load_component_by_name(component_name,  # type: Text
                           model_dir,  # type: Text
                           metadata,  # type: Metadata
                           cached_component,  # type: Optional[Component]
                           **kwargs  # type: **Any
                           ):
    # type: (...) -> Optional[Component]
    """Resolves a component and calls its load method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.load(model_dir, metadata, cached_component, **kwargs)


def create_component_by_name(component_name, config):
    # type: (Text, RasaNLUModelConfig) -> Optional[Component]
    """Resolves a component and calls it's create method to init it based on a
    previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.create(config)
