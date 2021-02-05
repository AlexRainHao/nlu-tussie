'''
Useg: Entities handler of merging and filtering, including
        1. Entities merging and filtering for a certain extractor dynamicly
        2. Forward entities alignment after pipelines done
        3. Backward entities alignment after pipelines done
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any
from typing import Dict
from typing import List
from typing import Text

import os
import sys
import re
from copy import deepcopy
from collections import defaultdict
import logging
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)


from ner_yjcloud.tokenizers import Token, SegPos, PreEntity
from ner_yjcloud.components import Component
from ner_yjcloud.training_data import Message

logger = logging.getLogger(__name__)


class EntityExtractor(Component):
    def add_extractor_name(self, entities):
        # type: (List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
        for entity in entities:
            entity["extractor"] = self.name
        return entities

    def add_processor_name(self, entity):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]

        return entity

    @staticmethod
    def find_entity(ent, text, tokens):
        offsets = [token.offset for token in tokens]
        ends = [token.end for token in tokens]

        if ent["start"] not in offsets:
            message = ("Invalid entity {} in example '{}': "
                       "entities must span whole tokens. "
                       "Wrong entity start.".format(ent, text))
            raise ValueError(message)

        if ent["end"] not in ends:
            message = ("Invalid entity {} in example '{}': "
                       "entities must span whole tokens. "
                       "Wrong entity end.".format(ent, text))
            raise ValueError(message)

        start = offsets.index(ent["start"])
        end = ends.index(ent["end"]) + 1
        return start, end

    def filter_trainable_entities(self, entity_examples):
        # type: (List[Message]) -> List[Message]
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than self.name (e.g. 'ner_crf')
        are removed."""

        filtered = []
        for message in entity_examples:
            entities = []
            for ent in message.get("entities", []):
                extractor = ent.get("extractor")
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            data['entities'] = entities
            filtered.append(
                Message(text=message.text,
                        data=data,
                        output_properties=message.output_properties,
                        time=message.time))

        return filtered



class EntityObj(object):
    """A stack entity used for merging or filtering"""
    def __init__(self, start_id = float("inf"),
                 end_id = 0,
                 text = '',
                 entity = None,
                 confidence = 1.):
        self.start_id = start_id
        self.end_id = end_id
        self.text = text
        self.entity = entity
        self.confidence = confidence

    def set_from_pre_entity(self, start_id, end_id, text, entity, confidence = 1.):
        self.start_id = min(start_id, self.start_id)
        self.end_id = max(end_id, self.end_id)
        self.text += text
        self.entity = entity
        self.confidence = confidence if not self.confidence else 0.5 * (self.confidence
                                                                        + confidence)

    def set_from_conjunction(self, text):
        self.end_id += len(text)
        self.text += text

    def dump_to_result(self):
        return {"start": self.start_id,
                "end": self.end_id,
                "value": self.text,
                "entity": self.entity,
                "confidence": self.confidence}


class ConjuncEntityDynamic():
    """
    Entities merging or filtering for a certain extractor. Supports
    1. Entities filter or merge by regex pattern 
    2. Entities merge by special token character
    3. Entity label mapping
    
    Usage examples
    1. For spacy extracotr or others similar
        suppose we have obtained `tokens`, `pos`, and `pre_entities` from spacy pipeline
            >>> text = "罗翔和张三是朋友关系，2020年3月通过网络贷款平台认识，借款金额为3000元整"
            >>> tokens, pos, entities = spacy_tokenizer(text, nlp)
        
        then define the dynamic post-progressing handler as follows
            >>> a = ConjuncEntityDynamic(tokens, entities, pos.get("pos"))
            >>> a.register(0, src = ["PERSON", "PER"], tar = "PER")
            >>> a.register(0, src = ["FAC", "LOC", "GPE"], tar = "LOC")
            >>> a.register(0, src = ["ORG", "NORP"], tar = "ORG")
            >>> a.register(0, src = ["MONEY"], tar = "Money")
            >>> a.register(0, src = ["DATE", "TIME"], tar = "Date")
            >>> a.register(1, src = ['SPACE'])
            >>> a.register(2, src = "LOC", tar = [["[门幢楼栋室巷屯乡镇元层区]$", "conj"]])
            >>> a.register(2, src = "PER", tar = [["[\da-zA-Z]", "clear"]])
            >>> a.register(2, src = "Date", tar = [["[0-9一二三四五六七八九十零]+岁", "clear"]])
            >>> res = a.main(raw_entity = [])
            >>> print(res)
            >>> [{'start': 0, 'end': 2, 'value': '罗翔', 'entity': 'PER', 'confidence': 1.0}, 
                 {'start': 3, 'end': 5, 'value': '张三', 'entity': 'PER', 'confidence': 1.0}, 
                 {'start': 11, 'end': 18, 'value': '2020年3月', 'entity': 'Date', 'confidence': 0.8301918513883276}, 
                 {'start': 34, 'end': 39, 'value': '3000元', 'entity': 'Money', 'confidence': 1.0}]

    2. For lac extractor or others similar
        since `lac_extractor` independent from the other pipelines, and its specific received parameters
        wo need a set of actions to do alignment as follows
        suppose we have obtained `tokens` and `pre_entities` by LAC firstly
            >>> text = "罗翔和张三是朋友关系，2020年3月通过网络贷款平台认识，借款金额为3000元整"
            >>> tokens, entities = lac.run(text)
        
        then define the dynamic pos-progressing handler as follows
            >>> a = ConjuncEntityDynamic(tokens, entities, initialConf = 1.)
            >>> a.convert_token()

            >>> a.register(0, src = ["PERSON", "PER"], tar = "PER")
            >>> a.register(0, src = ["FAC", "LOC", "GPE"], tar = "LOC")
            >>> a.register(0, src = ["ORG", "NORP"], tar = "ORG")
            >>> a.register(0, src = ["MONEY"], tar = "Money")
            >>> a.register(0, src = ["DATE", "TIME"], tar = "Date")
            >>> a.register(1, src = ['SPACE'])
            >>> a.register(2, src = "LOC", tar = [["[门幢楼栋室巷屯乡镇元层区]$", "conj"]])
            >>> a.register(2, src = "PER", tar = [["[\da-zA-Z]", "clear"]])
            >>> a.register(2, src = "Date", tar = [["[0-9一二三四五六七八九十零]+岁", "clear"]])
            >>> res = a.main(raw_entity = [])
            >>> print(res)
            >>> [{'start': 0, 'end': 2, 'value': '罗翔', 'entity': 'PER', 'confidence': 1.0}, 
                 {'start': 3, 'end': 5, 'value': '张三', 'entity': 'PER', 'confidence': 1.0}, 
                 {'start': 11, 'end': 18, 'value': '2020年3月', 'entity': 'Date', 'confidence': 1.0}, ]
    """
    def __init__(self, tokens, pre_entity, pos = None, threShold = .7, initialConf = 1.):
        if not pos:
            pos = deepcopy(pre_entity)

        self.tokens = tokens
        self.pre_entity = pre_entity
        self.pos = pos

        self.threShold = threShold
        self.initialConf = initialConf

        self.mappingLabel = defaultdict(str) # {"PER": "PER"}
        self.conjuncLabel = [] # ["SPACE"]

        self.conjPattern = defaultdict(list)
        self.clearPattern = defaultdict(list)


    @property
    def entities(self):
        if self.pos:
            return self.merge_pos_entity(self.pos, self.pre_entity)


    def convert_token(self):
        """
        convert tokens and pre-entities obatined from extractor 
        like `lac extractor` to `Token` and `PreEntity` format
        
        E.X.
            Input:  tokens: ["罗翔", "和", "张三",...,]
                    pre_entity: ["PER", "CONJ", "PER",...,]

            Output: tokens: [{"text":"罗翔", "offset":0, "end": 2, "data": {}},
                             {"text":"和", "offset":2, "end": 3, "data": {}},
                             {"text":"张三", "offset":3, "end": 5, "data": {}},
                             ...]
                    pre_entity: [{"text": 罗翔, "start_id":0, "end_id": 2, "eneity": "PER", "confidence": 1.},
                                 {"text": 和, "start_id":2, "end_id": 3, "eneity": "CONJ", "confidence": 1.},
                                 {"text": 张三, "start_id":3, "end_id": 5, "eneity": "PER", "confidence": 1.},
                                 ...]
        """

        tokens, entities = [], []
        start_id = 0

        for idx, (token, entity) in enumerate(zip(self.tokens, self.pre_entity)):
            _tk = Token(offset = start_id, text = token)

            _pe = PreEntity(start_id = _tk.offset, end_id = _tk.end,
                            text = token, entity = entity, confidence = self.initialConf)

            tokens.append(_tk)
            entities.append(_pe)

        self.tokens = tokens
        self.pre_entity = entities


    def register(self, method, src = None, tar = None):
        """a public register method for each configuration"""
        if method == 0:
            # for register mapping_label
            self._register_mapping_label(tar, src)

        elif method == 1:
            # for register conjucntion label

            labels = src or tar

            self._register_conjunc_label(labels)

        elif method == 2:
            # for register regex patterns

            self._register_regex_pattern(src, tar)

        else:
            pass

    def _register_mapping_label(self, tar_label, src_label):
        """
        mapping a source label obatained from pre-entities to target label
        
        This method handle two main condition from other extractors experience
            Cond.1  for 1-tar to N-src, E.X. spacy extractor
            Cond.2  for 1-tar to 1-src, E.X. lac extractor

        the main difference from these conditions is that whether receieve a src list and 
        convert it to a 1-tar to 1-src dictionary
        
        E.X.
            Input:  tar: "PER"
                    src: ["PERSON", "PER"]
            Output:
                    {"PERSON": "PER", "PER": "PER"}
        
        """
        if isinstance(src_label, list):
            for s_l in src_label:
                self.mappingLabel[s_l] = tar_label


        elif isinstance(src_label, str):
            self.mappingLabel[src_label] = tar_label

        else:
            raise TypeError("only received list or str parameter")


    def _register_conjunc_label(self, c_label):
        """register source conjuction labels"""
        if isinstance(c_label, list):
            pass

        elif isinstance(c_label, str):
            c_label = [c_label]

        self.conjuncLabel.extend(c_label)


    def _register_regex_pattern(self, src_label, patterns):
        """register a set of regex patterns
        the patterns have a format as list, of which like [pattern, flag]
        the instance `flag` as a role of leading different progressing,
        
        E.X.
            flag = "conj", for conjunction like LOC
            flag = "clear", for justification whether a extracted entity is legal
        """
        if not (isinstance(patterns, list) and isinstance(patterns[0], list)):
            raise ValueError("`patterns` received 2-D list")

        for pat in patterns:
            if pat[1] not in ["conj", "clear"]:
                logger.warning(f"only support for `conj` or `clear` actions, and {pat[0]} would ignored")
                continue

            elif pat[1] == "conj":
                self.conjPattern[src_label].append([re.compile(pat[0]), pat[1]])

            elif pat[1] == "clear":
                self.clearPattern[src_label].append([re.compile(pat[0]), pat[1]])



    @staticmethod
    def merge_pos_entity(pos: List[str], entity: List[PreEntity]) -> List[PreEntity]:
        """merge to one instance from pos and pre_entities
        
        Since of different input received on entity progressing between different extractors,
        (e.g. confilict between `spacy_extractor` and `lac_extractor`)
        and on consideration for a simplifed method to conduct post-progressing.
        This method make the pos infomation saved in pos to pre_entity with no entity label,
        then use entity label in pre_entity to mapping entity from mapping label dictionary only.

        E.X.
            Input: pos: ["Noun", "CCONJ", "Noun",...,]
                   entity:  [{"text":"罗翔", "start_id":0, "end_id": 2, "eneity": "PER"},
                             {"text":"和", "start_id":None, "end_id": None, "eneity": "None"},
                             {"text":"张三", "start_id":3, "end_id": 5, "eneity": "PER"},
                             ...]
            Output: entity: [{"text":"罗翔", "start_id":0, "end_id": 2, "eneity": "PER"},
                             {"text":"和", "start_id":None, "end_id": None, "eneity": "CCONJ"},
                             {"text":"张三", "start_id":3, "end_id": 5, "eneity": "PER"},
                             ...]
        """
        for idx, (x, y) in enumerate(zip(pos, entity)):
            if y.entity:
                continue

            else:
                entity[idx].entity = x

        return entity


    def clear_regex_search(self, label, text):
        """a method for certain label justification whether clear"""
        pattern = self.clearPattern.get(label, [])

        if not pattern:
            return False

        s_f = False

        for pat, _ in pattern:
            s_f |= True if pat.search(text) else False

        return s_f

    def conj_regex_search(self, label, text):
        """a method for certain label justification whether conj"""
        pattern = self.conjPattern.get(label, [])

        if not pattern:
            return False

        s_f = False

        for pat, _ in pattern:
            s_f |= True if pat.search(text) else False

        return s_f


    def confidence_handler(entity: EntityObj, pre_entity: PreEntity,
                           raw_entities: List = []):
        """only received entity over given confidence threshold"""

        pass


    def initialize_entity_stack(self, *args, **kwargs):
        curr_entity = EntityObj(confidence = self.initialConf)

        if args:
            curr_entity.set_from_pre_entity(*args)

        elif kwargs:
            curr_entity.set_from_pre_entity(*kwargs.values())

        return curr_entity


    def main(self, raw_entity):
        """
        A dynamic entity post-progressing method
        used for single extractor to handle conditions as follows:
            1. merge consecutive entities with same label
            2. clear entities with given regex feature
            3. dynamic update or change label through given regex feature
            4. merge entities with given conjunction token
        
        Returns
        -------
        finally expected json format ner results
        
        TODO:
            back-off method for condition 4. if it's entity satisfy condition 2.
        """
        # initialize
        curr_entity = EntityObj(confidence = self.initialConf)
        start_id = 0
        end_id = 0

        for token, entity in zip(self.tokens, self.entities):

            end_id += len(token.text)

            entity.entity = self.mappingLabel.get(entity.entity, None)

            # same entity
            if entity.entity == curr_entity.entity and entity.entity:
                curr_entity.set_from_pre_entity(start_id,
                                                end_id,
                                                token.text,
                                                entity.entity,
                                                entity.confidence)


            # encounder a conjunction
            # TODO: back-off handler
            elif entity.entity in self.conjuncLabel and curr_entity.entity:
                curr_entity.set_from_conjunction(token.text)


            # for regex pattern
            # search begin from `conj` to `clear`
            # for conj regex pattern
            elif curr_entity.entity in self.conjPattern.keys():
                pat = self.conjPattern.get(curr_entity.entity)

                s_f = self.conj_regex_search(curr_entity.entity,
                                             curr_entity.text + token.text)

                if s_f:
                    curr_entity.set_from_conjunction(token.text)

                elif curr_entity.confidence >= self.threShold:
                    raw_entity.append(curr_entity.dump_to_result())
                    curr_entity = EntityObj(confidence = self.initialConf)

                else:
                    curr_entity = self.initialize_entity_stack(start_id = start_id,
                                                               end_id = end_id,
                                                               text = token.text,
                                                               enity = entity.entity,
                                                               confidence = entity.confidence)


            # for clear regex pattern
            elif curr_entity.entity in self.clearPattern.keys():

                s_f = self.clear_regex_search(curr_entity.entity,
                                              curr_entity.text)

                if not s_f and curr_entity.confidence >= self.threShold:
                    raw_entity.append(curr_entity.dump_to_result())


                curr_entity = self.initialize_entity_stack(start_id = start_id,
                                                           end_id = end_id,
                                                           text = token.text,
                                                           enity = entity.entity,
                                                           confidence = entity.confidence)

            # for a new entity
            else:
                if curr_entity.entity and curr_entity.confidence >= self.threShold:
                    raw_entity.append(curr_entity.dump_to_result())

                curr_entity = self.initialize_entity_stack(start_id = start_id,
                                                           end_id = end_id,
                                                           text = token.text,
                                                           enity = entity.entity,
                                                           confidence = entity.confidence)

            start_id += len(token.text)

        # for remain `EntityObj` during iteration
        if curr_entity.entity:

            s_f = self.clear_regex_search(curr_entity.entity, curr_entity.text)

            if not s_f and curr_entity.confidence >= self.threShold:
                raw_entity.append(curr_entity.dump_to_result())

        return raw_entity


class PostProcessor():
    """
    Merge entities with same start pointer or end pointer
    
    E.X.
        >>> a = PostProcessor([])
        >>> a.test_case()
        
        or suppose we haved obatined ner results represented as `res`
        >>> a = PostProcessor.pipeline(res)
        >>> print(a)

    """

    @staticmethod
    def _if_cat_forward(src_entity, tar_entity):
        if src_entity[0] <= tar_entity["end"] and src_entity[2] == tar_entity["entity"]:
            return [tar_entity["end"],
                    tar_entity["value"],
                    tar_entity["entity"],
                    tar_entity["confidence"]]

        else:
            return

    @staticmethod
    def _if_cat_backward(src_entity, tar_entity):
        if src_entity[0] >= tar_entity["start"] and src_entity[2] == tar_entity["entity"]:
            return [tar_entity["start"],
                    tar_entity["value"],
                    tar_entity["entity"],
                    tar_entity["confidence"]]

        else:
            return


    def forwardProcessor(self, entities):
        """merge entities with same entity label and start pointer"""

        if not entities:
            return entities

        mapping_pointer = {}

        for entity in entities:
            src_entity = mapping_pointer.get(str(entity["start"]) + ' ' + entity["entity"], [])

            if src_entity:
                tar_entity = self._if_cat_forward(src_entity, entity)

                if tar_entity:
                    mapping_pointer[str(entity['start']) + ' ' +entity['entity']] = [entity["end"],
                                                                                     entity["value"],
                                                                                     entity["entity"],
                                                                                     entity["confidence"]]
                else:
                    continue


            else:
                mapping_pointer[str(entity["start"]) + ' ' +entity["entity"]] = [entity["end"],
                                                                                 entity["value"],
                                                                                 entity["entity"],
                                                                                 entity["confidence"]]

        mapping_pointer = sorted(mapping_pointer.items(), key = lambda x: int(x[0].split(' ')[0]))

        return [EntityObj(int(x[0].split(' ')[0]), x[1][0], x[1][1], x[1][2], x[1][3]).dump_to_result() for x in mapping_pointer]


    def backwardProcessor(self, entities):
        """merge entities with same entity label and end pointer"""

        if not entities:
            return entities

        mapping_pointer = {}

        for entity in entities:
            src_entity = mapping_pointer.get(str(entity["end"]) + ' ' + entity["entity"], [])

            if src_entity:
                tar_entity = self._if_cat_backward(src_entity, entity)

                if tar_entity:
                    mapping_pointer[str(entity['end']) + ' ' + entity['entity']] = [entity["start"],
                                                                                   entity["value"],
                                                                                   entity["entity"],
                                                                                   entity["confidence"]]
                else:
                    continue


            else:
                mapping_pointer[str(entity["end"]) + ' ' + entity["entity"]] = [entity["start"],
                                                                               entity["value"],
                                                                               entity["entity"],
                                                                               entity["confidence"]]

        mapping_pointer = sorted(mapping_pointer.items(), key = lambda x: int(x[0].split(' ')[0]))

        return [EntityObj(x[1][0], int(x[0].split(' ')[0]), x[1][1], x[1][2], x[1][3]).dump_to_result() for x in mapping_pointer]

    def uniqueProcessor(self):
        """pass"""
        pass

    @classmethod
    def pipeline(cls, entities):
        res = cls.forwardProcessor(cls, entities)
        res = cls.backwardProcessor(cls, res)

        return res


    def test_case(self):
        res = [{'start': 0, 'end': 2, 'value': '罗翔', 'entity': 'PER', 'confidence': 1.0},
               {'start': 3, 'end': 5, 'value': '张三', 'entity': 'PER', 'confidence': 1.0},
               {'start': 0, 'end': 5, 'value': '罗翔和张三', 'entity': 'PER', 'confidence': 1.0},
               {'start': 16, 'end': 22, 'value': '杭州市余杭区', 'entity': 'ORG', 'confidence': 1.0},
               {'start': 16, 'end': 32, 'value': '杭州市余杭区西溪北苑北区109幢', 'entity': 'LOC', 'confidence': 1.0},
               {"start": 22, "end": 32, "value": "西溪北苑北区109幢", "entity": "LOC", "confidence": 1.0},
               {'start': 45, 'end': 50, 'value': '5205元', 'entity': 'Money', 'confidence': 1.0},
               {'start': 45, 'end': 55, 'value': '12132.5205元', 'entity': 'Money', 'confidence': 1.0},
               ]

        res_1 = self.forwardProcessor(res)
        res_2 = self.backwardProcessor(res_1)

        print(res_2)


class tagNormalizer:
    """
    Filter entities with illegal tagging patterns

    E.X
    S_I -> B_I
    S_E -> B_E
    B_S -> B-I
    O_I ->O_O / B_I (determined by length of I ?)

    et.al
    """

    def __init__(self, tags, window = 2, use_default = True):
        self.tags = tags
        self.window = window

        self.handlers = [self._defaults_handler] if use_default else []

        self.names = [self._default_name] if use_default else []

    @property
    def _default_name(self):
        return "S_I -> B_I, S_E -> B_E, O_I -> O_O/B_I, B_S -> B_I"

    def _check_poster(self, tags, idx):
        non_O_count = len(list(filter(
            lambda x: x.startswith("O"),
            tags[(idx + 1): (idx + self.window + 1)]
        )))
        return True if non_O_count else False

    def _defaults_handler(self, tags):
        """pass"""

        tags = tags + ["O"] * max((self.window - 1), 0)

        if len(set(tags)) == 1:
            return tags

        for idx in range(len(tags) - 1):
            c_t = tags[idx]
            n_t = tags[idx + 1]

            if c_t.startswith("S") and n_t.startswith("I"):
                if c_t[1:] == n_t[1:]:
                    tags[idx] = "B" + c_t[1:]

                elif self._check_poster(tags, idx):
                    tags[idx + 1] = "B" + n_t[1:]

                else:
                    tags[idx + 1] = "O"

            elif c_t.startswith("S") and n_t.startswith("E"):
                if c_t[1:] == n_t[1:]:
                    tags[idx] = "B" + c_t[1:]

                elif self._check_poster(tags, idx):
                    tags[idx + 1] = "B" + n_t[1:]

                else:
                    tags[idx + 1] = "O"

            elif c_t.startswith("B") and n_t.startswith("S"):
                if c_t[1:] == n_t[1:]:
                    tags[idx + 1] = "I" + n_t[1:]

            elif c_t.startswith("O") and n_t.startswith("I"):
                non_O_count = len(list(filter(
                    lambda x: x.startswith("O"),
                    tags[(idx + 1): (idx + self.window + 1)]
                )))

                if non_O_count > 1:
                    tags[idx] = "B" + c_t[1:]
                else:
                    tags[idx + 1] = "O"
        return tags

    def register_self_handler(self, method, name):
        self.handlers.append(method)
        self.names.append(name)

    def run(self):
        tags = self.tags.copy()

        for handler in self.handlers:
            try:
                tags = handler(tags)

            except:
                pass

        return tags
