"""
a series of method to obtained training dataset features, including
    * embedding method features
    * bert method features
    * segment features
    * raw text normalizers [have moved to normalizer module]

"""

from typing import Any, Dict, List, Text, Tuple, Callable, Union

import os
import os.path as opt
import logging

import re
from collections import namedtuple
import numpy as np

from .vocab import Vocab, Tags

logger = logging.getLogger(__name__)

__all__ = ["InputExample", "InputFeature",
           "softmax", "prepare_dataset", "prepare_bert_dataset",
           "get_seg_features", "pred_result_to_json"]

class InputExample(object):
    """a single set of examples of data,
    only used by `Bert` Processor
    """

    def __init__(self, guid = None, text = None, label = None):
        """

        Parameters
        ----------
        guid: Text
        text: Optional[Text, List[Text]]
        label: Optional[Text, List[Text]]
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeature(object):
    """A single set of features of data,
    only used by `Bert` Processor"""

    def __init__(self, input_ids, input_mask, input_type_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_ids = label_ids

def softmax(x):
    """softmax method"""

    r = np.exp(x - np.max(x))
    return r / r.sum(axis=-1, keepdims=True)

def prepare_dataset(examples, vocabs, tags, use_seg = True) -> List[namedtuple]:
    """
    obtain training dataset features for `embedding` method,
    where token vocab constructed by given examples,

    each example or sentence would extracted features including
        * token ids
        * segment ids, or word ids
        * label ids,

    and wrapper by a `namedtuple` way

    Parameters
    ----------
    examples: List[namedtuple]
    vocabs: Vocab
    tags: Tags
    use_seg : Bool
    lower: if use lower case

    Returns
    -------

    """
    scopes = namedtuple("example", ["text",
                                    "token_ids",
                                    "seg_ids",
                                    "tag_ids"])

    def prepare_sentence(ds, vocabs, tags, use_seg = True, showing = False):
        """pass"""
        tokens = ds.text
        words = ds.words
        token_ids = vocabs.get_sequence_id(tokens)
    
        seg_ids = [0] * len(tokens)
        if use_seg:
            seg_ids = get_seg_features(words)
    
        tag_ids = tags.get_taguence_id(ds.tags, update = False)
        
        # token_ids = [vocabs.UNK] + token_ids + [vocabs.UNK]
        # seg_ids = [0] + seg_ids + [0]
        # tag_ids = [tags.startIdx()] + tag_ids + [tags.stopIdx()]

        if showing:
            logger.info("*** Example Feature ***")
            logger.info(f"tokens:\t{tokens}")
            logger.info(f"token_ids:\t{token_ids}")
            logger.info(f"seg_ids:\t{seg_ids}")
            logger.info(f"tag_ids:\t{tag_ids}")
            
        
        return scopes(text = ''.join(tokens),
                      token_ids = token_ids,
                      seg_ids = seg_ids,
                      tag_ids = tag_ids)

    features = []
    for id, exam in enumerate(examples):
        if id == 1:
            features.append(prepare_sentence(exam, vocabs, tags, use_seg, True))
        else:
            features.append(prepare_sentence(exam, vocabs, tags, use_seg, False))
            
    return features

def prepare_bert_dataset(examples,
                         tokenizer,
                         max_seq_len,
                         tags):
    """
    obtain training dataset features for `bert` method,
    it is different from `embedding` way that
        the token vocab had constructed by Bert model,
        and segment ids features not be used

    features extracted including
        * token ids
        * mask ids
        * type ids

    as the format of single sentence features but pairs of Bert

    Parameters
    ----------
    examples: List[InputExample]
    tokenizer: ClassVar
    max_seq_len: Int,
    tags: Tags

    Returns
    -------

    """

    def prepare_sentence(example, tokenizer, max_seq_len, tags, showing = False):
        """pass"""
        guid = example.guid
        tokens = example.text
        labels = example.label
        lattice_tokens, lattice_labels = [], []

        for i, tok in enumerate(tokens):
            _tok = tokenizer.tokenize(tok)

            lattice_tokens.extend(_tok)

            for m in range(len(_tok)):
                if m == 0:
                    lattice_labels.append(labels[i])
                else:
                    lattice_labels.append("O")

        # truncate sequence
        lattice_tokens = [tags.START] + lattice_tokens[:max_seq_len - 2] + [tags.STOP]

        lattice_labels = [tags.START] + \
                         lattice_labels[:max_seq_len - 2] + \
                         [tags.STOP]

        # get features
        token_ids = tokenizer.convert_tokens_to_ids(lattice_tokens)
        label_ids = tags.get_taguence_id(lattice_labels, update = False)
        mask_ids = [1] * len(token_ids)
        type_ids = [0] * len(token_ids)

        # padding
        while len(token_ids) < max_seq_len:
            token_ids.append(0)
            mask_ids.append(0)
            type_ids.append(0)
            label_ids.append(tags.stopIdx())

        assert len(token_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(mask_ids) == max_seq_len
        assert len(type_ids) == max_seq_len

        if showing:
            logger.info("*** Example Feature ***")
            logger.info(f"guid:\t{guid}")
            logger.info(f"tokens:\t{lattice_tokens}")
            logger.info(f"labels:\t{lattice_labels}")
            logger.info(f"token_ids:\t{token_ids}")
            logger.info(f"label_ids:\t{label_ids}")
            logger.info(f"mask_ids:\t{mask_ids}")
            logger.info(f"type_ids:\t{type_ids}")

        return InputFeature(input_ids = token_ids,
                            input_mask = mask_ids,
                            input_type_ids = type_ids,
                            label_ids = label_ids)

    # return [prepare_sentence(exam, tokenizer, )]
    features = []
    for id, exam in enumerate(examples):
        if id == 1:
            features.append(prepare_sentence(exam, tokenizer, max_seq_len, tags, showing = True))

        else:
            features.append(prepare_sentence(exam, tokenizer, max_seq_len, tags, showing = False))

    return features

def get_seg_features(words):
    """
    obtain training dataset segment features, where
        0 -> a single word
        1 -> word start
        2 -> middle part
        3 -> word tail

    E.X.
        text = ["嘿", "今天", "天气", "怎么样"]
        feature = [0, 1, 3, 1, 3, 1, 2, 3]

    Parameters
    ----------
    words: List[Text]

    Returns
    -------

    """
    seg_features = []

    for word in words:
        if len(word) == 1:
            seg_features.append(0)

        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_features.extend(tmp)

    return seg_features


def _create_entity_dict(entity_name,
                        entity_start,
                        entity_end,
                        entity_tag,
                        confidence):
    return {
        'start': entity_start,
        'end': entity_end,
        'value': entity_name,
        'entity': entity_tag,
        'confidence': confidence
    }

def pred_result_to_json(text: Union[Text, List[Text]],
                        pred_tags: List[Text],
                        confidence: List) -> Any:
    """convert entities extracted from model to `Extractor pipeline` output legal format"""

    if isinstance(text, List):
        text = ''.join(text)

    entities = []

    entity_name = ''
    entity_tag = ''
    entity_start = 0
    score = []

    for idx, (char, tag, cf) in enumerate(zip(text, pred_tags, confidence)):
        if tag[0] == "S":
            if entity_name:
                entities.append(
                    _create_entity_dict(entity_name, entity_start, idx, entity_tag, max(score))
                )
                entity_name = ''
                entity_tag = ''
                score = []

            entities.append(
                _create_entity_dict(char, idx, idx + 1, tag[2:], cf)
            )

        elif tag[0] == "B":
            if entity_name:
                entities.append(
                    _create_entity_dict(entity_name, entity_start, idx, entity_tag, max(score))
                )
                entity_name = ''
                entity_tag = ''
                score = []

            entity_name += char
            entity_tag = tag[2:]
            score.append(cf)
            entity_start = idx

        elif tag[0] == "I":
            entity_name += char

        elif tag[0] == "E":
            entities.append(
                _create_entity_dict(entity_name, entity_start, idx, entity_tag, max(score))
            )
            entity_name = ''
            entity_tag = ''
            score = []

        else:
            if entity_name:
                entities.append(
                    _create_entity_dict(entity_name, entity_start, idx, entity_tag, max(score))
                )

            entity_name = ''
            entity_tag = ''
            entity_start = idx
            score = []
            
    if entity_name:
        entities.append(
            _create_entity_dict(entity_name, entity_start, len(text), entity_tag, max(score))
        )

    return entities
