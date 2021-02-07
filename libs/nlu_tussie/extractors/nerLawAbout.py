# -*- coding: utf-8 -*-
'''
Usage: Regular pattern, window search from model parsed results NER for

    * Role -- 原告、被告、原告代理人、被告代理人
    * Court -- 基层法院、中级法院、基层人民法院、最高人民法院、湖州中院、杭州高院
    * Legal representative

But Dependency analysis or model based approach worth trying

Noticed this method recommended used followed from `LAC` or `SPACY` or others models
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import jieba.posseg as psg
from copy import deepcopy

from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from nlu_tussie.extractors import EntityExtractor
from nlu_tussie.model import Metadata
from nlu_tussie.training_data import Message
from nlu_tussie.training_data import TrainingData



def _gold_window_forward_search(gold_format, gold_label, token_word, entities, window_size=1):
    """forward key word search, suggest a window size equal to 1"""
    res = []

    token_spx = {x.offset: (idx, x.text) for idx, x in enumerate(token_word)}
    token_rev_spx = {idx: (offset, text) for offset, (idx, text) in token_spx.items()}

    ent_spx = {x["start"]: x["value"] for x in entities if x["entity"] in gold_label}

    for e_s_point, e_text in ent_spx.items():
        try:
            e_s_idx = token_spx.get(e_s_point)[0]

            _s_idx = max(0, e_s_idx - window_size - 1)
            # _e_idx = min(list(token_rev_spx.keys())[-1], e_s_idx + window_size + 1)

            # forward
            for _id in range(_s_idx, e_s_idx):
                _w_text = token_rev_spx[_id][1]

                if _w_text in gold_format:
                    w_text = ''.join([token_rev_spx[x][1] for x in range(_id, e_s_idx)]) + e_text
                    res.append((w_text, token_rev_spx[_id][0], token_rev_spx[_id][0] + len(w_text)))
                    break
        except:
            pass

    return res


def _glod_window_backward_search(gold_format, gold_label, entities, window_size):
    """backward key word search"""
    res = []
    return res


class LawAboutExtractor(EntityExtractor):
    name = 'law_about_extractor'

    provides = ['entities']

    requires = ['tokens', 'pos', "entities"]

    defaults = {"ROLE": "ROLE",
                "COURT": "COURT",
                "LEGAL_REPRESENTIVE": "LEGAL_REPRESENTIVE",
                "TITLE": "TITLE"}

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(LawAboutExtractor, self).__init__(component_config)
        # self.tagName = {"ROLE": "ROLE",
        #                 "COURT": "COURT",
        #                 "LEGAL_REPRESENTIVE": "LEGAL_REPRESENTIVE",
        #                 "TITLE": "TITLE"}
        self.component_config = component_config
        self.windowSize = 2


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):  # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.searchFunc(message))

        message.set("entities", extracted, add_to_output = True)


    @classmethod
    def load(cls,
             model_dir=None,   # type: Optional[Text]
             model_metadata=None,   # type: Optional[Metadata]
             cached_component=None,   # type: Optional[Component]
             **kwargs  # type: **Any
             ):  # type: (...) -> Component

        meta = model_metadata.for_component(cls.name)

        return cls(meta)


    @staticmethod
    def _handle_role(token_word, token_pos, entities, window_size, **kwargs):

        # string = ''.join(token_word)
        #
        # gold_format = ["原告", "被告", "原告代理人", "被告代理人", "上诉人", "被上诉人",
        #                "承办人", "代理人", "诉讼人", "当事人"]
        #
        # res = []
        # for role in gold_format:
        #     search_res = re.finditer("(%s)" % role, string)
        #
        #     for _search in search_res:
        #         res.append((_search.group(1), _search.span()[0], _search.span()[1]))
        #
        # return res

        gold_format = ["原告", "被告", "原告代理人", "被告代理人", "上诉人", "被上诉人",
                       "承办人", "代理人", "诉讼人", "当事人"]

        gold_label = ["PERSON", "PER", "RESIDENT", "FAC", "GPE", "LOC", "ORG", "NORP"]

        res = _gold_window_forward_search(gold_format, 
                                          gold_label,
                                          token_word,
                                          entities,
                                          window_size)

        return res


    @staticmethod
    def _handle_court(token_word, token_pos, entities, **kwargs):
        """modification for extracting `Court` based on `ORG` entity"""

        pat_1 = re.compile("(第?[0-9一二三四五六七八九十]?[基中高最]?[层级高]?(人民)?法[院庭])")
        pat_2 = re.compile("([基中高]院)")

        res = []

        for entity in entities:
            _re_flag = pat_1.findall(entity["value"]) or pat_2.findall(entity["value"])

            if _re_flag:
                res.append((entity["value"], entity["start"], entity["end"]))

        # s_pot = 0
        # e_pot = 0
        # l_pot = 0
        # _curr_pointer = 0
        #
        # string = ''.join(token_word)
        #
        # _curr_string = deepcopy(string)
        #
        #
        # while _curr_string:
        #     tar_seg = None
        #     for pat in [pat_1, pat_2]:
        #         search_res = pat.search(_curr_string)
        #
        #         if search_res:
        #             s_pot, e_pot = search_res.span()
        #             tar_seg = search_res.group(0)
        #             break
        #
        #     if not tar_seg:
        #         return res
        #
        #     # ================
        #     if s_pot == 0:
        #         s_pot += l_pot
        #         e_pot += l_pot
        #         res.append((tar_seg, s_pot, e_pot))
        #         _curr_string = string[e_pot:]
        #         l_pot = deepcopy(e_pot)
        #         continue
        #         # return tar_seg, s_pot, e_pot
        #
        #     tag_list = list(map(lambda x: (x.word, x.flag), psg.cut(_curr_string[:s_pot])))
        #     # print('*' * 40)
        #     # print(tag_list)
        #
        #     _ns_pre_str = ''
        #     for tag in tag_list[::-1]:
        #         if tag[1] in ['u', 'uj', 'r', 'c', 'e', 'y', 'x', 'ns']:
        #             _ns_pre_str = tag[0] + _ns_pre_str
        #         else:
        #             break
        #     s_pot -= len(_ns_pre_str)
        #     tar_seg = _ns_pre_str + tar_seg
        #
        #     s_pot += l_pot
        #     e_pot += l_pot
        #     res.append((tar_seg, s_pot, e_pot))
        #     _curr_string = string[e_pot:]
        #     l_pot = deepcopy(e_pot)

        return res

    @staticmethod
    def _handle_legal_role(token_word, token_pos, entities, window_size, **kwargs):
        # res = []
        # s_pot = 0
        # e_pot = 0

        # tag_list = list(map(lambda x,y: (x,y), token_word, token_pos))

        # for id, tag in enumerate(tag_list):
        #     if tag[1] in ['NOUN', "PROPN"]:
        #         _curr_str = ''.join([x[0] for x in tag_list[max(0, id-window_size):(id+window_size)]])

        #         try:
        #             _pot = _curr_str.index("法人代表") or _curr_str.index('法人')
        #             e_pot = s_pot + len(tag[0])
        #             res.append((tag[0], s_pot, e_pot))
        #         except:
        #             pass

        #     s_pot += len(tag[0])

        # return res

        gold_format = ["法人代表", "法人"]

        gold_label = ["PERSON", "PER", "RESIDENT", "FAC", "GPE", "LOC", "ORG", "NORP"]

        res = _gold_window_forward_search(gold_format,
                                          gold_label,
                                          token_word,
                                          entities,
                                          window_size)

        return res

    @staticmethod
    def _handle_title(string):
        pass

    def searchFunc(self, example):

        raw_entity = example.get('entities', [])
        string = example.text

        if string.strip() == '':
            return raw_entity

        tokenized_word = example.get('tokens')
        tokenized_pos = example.get('pos').get('pos')

        for _s_name, _s_func in zip(['ROLE', 'COURT', 'LEGAL_REPRESENTIVE'],
                                    [self._handle_role, self._handle_court, self._handle_legal_role]):
            _s_res = _s_func(tokenized_word,
                             tokenized_pos,
                             raw_entity,
                             window_size = self.windowSize)

            if _s_res:
                for _res in _s_res:
                    raw_entity.append({'start': _res[1],
                                       'end': _res[2],
                                       'value': _res[0],
                                       'entity': self.component_config.get(_s_name,
                                                                           _s_name),
                                       "confidence": 1.})

        return raw_entity



# a = NumberPatternExtractor()
# target_list = a.searchFunc("我的手机号是吧那个是13684110597啊啊是13384615442的你说川BU512U是身份证号513902199511246711对的就是浙A05872这个")
# print(target_list)
