# -*- coding: utf-8 -*-
'''
Usage: Regular pattern and window search NER for

    * Age
    * Sex
    * Origin
    * Race

But Dependency analysis or model based approach worth trying
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
# import jieba.posseg as psg
from copy import deepcopy

from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_yjcloud.extractors import EntityExtractor
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message
from ner_yjcloud.training_data import TrainingData

class IdentityPatternExtractor(EntityExtractor):
    name = "identity_pattern_extractor"

    provides = ['entities']

    requires = ['pos']

    defaults = {"AGE": "Age",
                "GENDER": "Gender",
                "RACE": "Race",
                "ORIGIN": "Origin"}


    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(IdentityPatternExtractor, self).__init__(component_config)
        # self.tagName = {"AGE": "Age",
        #                 "GENDER": "Gender",
        #                 "RACE": "Race",
        #                 "ORIGIN": "Origin"}
        self.component_config = component_config

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.searchFunc(message))

        message.set("entities", extracted, add_to_output = True)


    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)

    @staticmethod
    def _handle_origin(token_word, token_pos, window_size=3):
        # 城乡标准区别
        # 1. **省**市**区
        # 2. **省**县**乡

        tag_list = list(map(lambda x,y: (x,y), token_word, token_pos))

        value_list, spointer_list, epointer_list = [], [], []

        if len(tag_list) <= 1:
            return value_list, spointer_list, epointer_list

        ns_count_list = []
        _curr_count = 0
        for tag in tag_list:
            if tag[1] in ['PROPN', 'NOUN']:
                _curr_count += 1
            else:
                _curr_count = 0
            ns_count_list.append(_curr_count)

        for id, count in enumerate(ns_count_list):
            _win_pot = max(0, id - window_size - 1)
            _win_string = ''.join(x[0] for x in tag_list[_win_pot:id])

            if count == 2 and id != len(ns_count_list) - 1 and ns_count_list[id + 1] != 3:
                _win_flag = True if list(filter(lambda x: x in _win_string, ['祖籍', '户籍', '籍贯'])) else False

                if tag_list[id][0][-1] in ['区', '乡', '镇', '县', '县城', '人'] and _win_flag:
                    res_value = ''.join(x[0] for x in tag_list[(id - 1):(id + 1)])
                    res_pointer = len(''.join(x[0] for x in tag_list[:(id - 1)]))

                    value_list.append(res_value)
                    spointer_list.append(res_pointer)
                    epointer_list.append(res_pointer + len(res_value))


            elif count == 3 and id != len(ns_count_list) - 1 and ns_count_list[id + 1] != 4:
                _win_flag = True if list(filter(lambda x: x in _win_string, ['祖籍', '户籍', '籍贯'])) else False

                if tag_list[id][0][-1] in ['区', '乡', '镇', '县', '县城', '人'] and _win_flag:
                    res_value = ''.join(x[0] for x in tag_list[(id - 2):(id + 1)])
                    res_pointer = len(''.join(x[0] for x in tag_list[:(id - 2)]))

                    value_list.append(res_value)
                    spointer_list.append(res_pointer)
                    epointer_list.append(res_pointer + len(res_value))

        # return res_value, res_pointer, res_pointer + len(res_value)
        return value_list, spointer_list, epointer_list

    @staticmethod
    def _handle_gender(token_word, token_pos):

        string = ''.join(token_word)

        pat = re.compile('((性别)?(男|女)性?)')
        # res_value = None
        # start_pointer = 0
        # end_pointer = 0

        res_value_list, spointer_list, epointer_list = [], [], []

        # search_res = pat.search(string)
        search_res = pat.finditer(string)

        # if search_res:
        for _search in search_res:
            res_value_list.append(_search.group(0))
            spointer_list.append(_search.span()[0])
            epointer_list.append(_search.span()[1])

        # return res_value, start_pointer, end_pointer
        return res_value_list, spointer_list, epointer_list


    @staticmethod
    def _handle_age(token_word, token_pos):

        string = ''.join(token_word)

        # pat_1 = re.compile('(今年[0-9一二三四五六七八九十百]+岁?)')
        pat_1 = re.compile("([0-9一二三四五六七八九十百]+个月大)")
        pat_2 = re.compile('([0-9一二三四五六七八九十百]+岁)')

        res_value, start_pointer, end_pointer = [], [], []

        search_pot = 0

        while search_pot < len(string):
            _curr_string = string[search_pot:]

            for pat in [pat_2, pat_1]:
                search_res = pat.search(_curr_string)

                if search_res:
                    res_value.append(search_res.group(1))
                    start_pointer.append(search_res.span()[0] + search_pot)
                    end_pointer.append(search_res.span()[1] + search_pot)

                    search_pot = search_res.span()[1] + search_pot

                    break

            if not search_res:
                break

        return res_value, start_pointer, end_pointer


    @staticmethod
    def _handle_race(token_word, token_pos):

        race_list = ["汉族","满族","蒙古族","回族","藏族","维吾尔族","苗族","彝族","壮族","布依族",
             "侗族,瑶族","白族","土家族","哈尼族","哈萨克族","傣族","黎族","傈僳族",
             "佤族,畲族","高山族","拉祜族","水族","东乡族","纳西族","景颇族","柯尔克孜族",
             "土族,达斡尔族","仫佬族","羌族","布朗族","撒拉族","毛南族",
             "仡佬族,锡伯族","阿昌族","普米族","朝鲜族","塔吉克族","怒族","乌孜别克族","俄罗斯族",
             "鄂温克族,德昂族","保安族","裕固族","京族","塔塔尔族","独龙族","鄂伦春族","赫哲族",
             "门巴族,珞巴族","基诺族"]

        res_value_list, s_pointer_list, e_pointer_list = [], [], []

        res_value = None
        start_pointer = 0
        end_pointer = 0

        tag_list = list(map(lambda x, y: (x, y), token_word, token_pos))

        for id, (word, pos) in enumerate(zip(token_word, token_pos)):
            #if pos == 'PROPN' and word.endswith('族'):
            if word in race_list:
                res_value = word
                end_pointer = start_pointer + len(word)

                res_value_list.append(res_value)
                s_pointer_list.append(start_pointer)
                e_pointer_list.append(end_pointer)

            start_pointer += len(word)


        return res_value_list, s_pointer_list, e_pointer_list


    def searchFunc(self, example):

        raw_entity = example.get("entities", [])
        string = example.text

        # tokenized_pos = example.get('pos')

        if string.strip() == '':
            return raw_entity

        tokenized_word = example.get('pos').get('word')
        tokenized_pos = example.get('pos').get('pos')

        for _s_name, _s_func in zip(['AGE', 'GENDER', 'RACE', "ORIGIN"],
                                    [self._handle_age, self._handle_gender, self._handle_race, self._handle_origin]):
            word_list, sp_list, ep_list = _s_func(tokenized_word,
                                                  tokenized_pos)

            if word_list:
                for _word, _sp, _ep in zip(word_list, sp_list, ep_list):
                    raw_entity.append({'start': _sp,
                                       'end': _ep,
                                       'value': _word,
                                       'entity': self.component_config.get(_s_name, _s_name),
                                       "confidence": 1.})

        return raw_entity


# a = IdentityPatternExtractor()
# text = ''
# res = a.searchFunc(text)
# print(res)
