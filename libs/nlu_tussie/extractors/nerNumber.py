# -*- coding: utf-8 -*-
'''
Usage: Regular pattern and window search NER for

    * ID Card
    * Phone Number
    * Bank Card
    * Case Number
    * Car License

But Dependency analysis or model based approach worth trying

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
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


class NumberPatternExtractor(EntityExtractor):
    name = "number_pattern_extractor"

    provides = ["entities"]

    requires = ['pos']

    defaults = {"idCard": "IDNumber", "phoneNum": "MobilePhone",
                "bankNum": "BankCard", "licePlat": "CarNumber",
                "caseNum": "CaseNum"}

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(NumberPatternExtractor, self).__init__(component_config)
        # self.tagName = {"idCard": "IDNumber", "phoneNum": "MobilePhone",
        #                 "bankNum": "BankCard", "licePlat": "CarNumber",
        #                 "caseNum": "CaseNum"}
        self.component_config = component_config

    def train(self, trainging_data, config, **kwargs):
        # type: (TrainingData) -> None

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
    def _handle_id_card(string):
        gold_format = ["11", "12", "13", "14", "15", "21", "22", "23", "31", "32",
                       "33", "34", "35", "36", "37", "41", "42", "43", "44", "45", "46",
                       "50", "51", "52", "53", "54", "61", "62", "63", "64", "65", "71",
                       "81", "82", "91"]

        # print(list(map(lambda x,y: text.replace(x,y), '一二', '12')))

        def cn2an(head_string):
            res_str = ''
            mapping = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                       '六': '6', '七': '7', '八': '8', '九': '9', '零': '0',
                       '幺': '1'}
            for _head in head_string:
                res_str += mapping.get(_head, _head)

            return res_str

        pat = re.compile("[^0-9幺一二三四五六七八九零]([0-9幺一二三四五六七八九零]{17}[0-9幺一二三四五六七八九零|X])[^0-9幺一二三四五六七八九零]")

        _res_flag = False
        _res = None  # string
        _span = None  # tuple

        string = ' ' + string + ' '
        _search_flag = pat.search(string)
        if _search_flag:
            _head_str = _search_flag.group(1)[:2]
            if _head_str in gold_format:
                _res_flag = True
                # _res = _search_flag.group(1)
                _res = str(_search_flag.group(1))
                _span = (_search_flag.span()[0], _search_flag.span()[1] - 2)

            elif cn2an(_head_str) in gold_format:
                _res_flag = True
                # _res = _search_flag.group(1)
                _res = str(_search_flag.group(1))
                _span = (_search_flag.span()[0], _search_flag.span()[1] - 2)

        # print(_res)

        return _res, _span, _res_flag, False


    @staticmethod
    def _handle_phone_number(string):
        pat = re.compile("[^0-9幺一二三四五六七八九零]([1一幺][345789三四五七八九][0-9幺一二三四五六七八九零]{9})[^0-9幺一二三四五六七八九零]")

        _res_flag = False
        _res = None
        _span = None
        _res_span = False

        string = ' ' + string + ' '
        _search_flag = pat.search(string)

        if _search_flag:
            _res_flag = True
            # _res = _search_flag.group(1)
            _res = _search_flag

            if _res.group(0) != _res.group(1):
                _res_span = True

            _res = str(_res.group(1))
            _span = (_search_flag.span()[0] - 1, _search_flag.span()[1] - 2)
        return _res, _span, _res_flag, _res_span


    @staticmethod
    def _handle_case_number(string):
        pat = re.compile("[^0-9幺一二三四五六七八九零]([\\(\\（]?[0-9幺一二三四五六七八九零]{4}[\\)\\）]?[最高法|京津冀晋蒙辽吉黑沪苏浙皖闽赣甘青宁新鲁豫鄂湘粤桂琼渝川黔滇藏陕台港澳][^年月日]+?[0-9幺一二三四五六七八九零]{0,4}\D+?[0-9幺一二三四五六七八九零]{1,4}号?)")

        _res_flag = False
        _res = None
        _span = None

        string = ' ' + string + ' '
        _search_flag = pat.search(string)

        if _search_flag:
            _res_flag = True
            # _res = _search_flag.group(1)
            _res = str(_search_flag.group(1))
            _span = (_search_flag.span()[0], _search_flag.span()[1] - 1)

        return _res, _span, _res_flag, False


    @staticmethod
    def _handle_licese_plat(string):
        pat = re.compile("([京津冀晋蒙辽吉黑沪苏浙皖闽赣甘青宁新鲁豫鄂湘粤桂琼渝川黔滇藏陕台港澳][A-Za-z][0-9幺一二三四五六七八九零A-Za-z]{5})")

        _res_flag = False
        _res = None
        _span = None

        string = ' ' + string + ' '
        _search_flag = pat.search(string)

        if _search_flag:
            _res_flag = True
            # _res = _search_flag.group(1)
            _res = str(_search_flag.group(1))
            _span = (_search_flag.span()[0] - 1, _search_flag.span()[1] - 1)

        return _res, _span, _res_flag, False

    @staticmethod
    def _handel_bank_number(string):
        pat = re.compile("[^0-9幺一二三四五六七八九零]([46四六]{1}[0-9零一二三四五六七八九幺]{18})[^0-9幺一二三四五六七八九零]")

        _res_flag = False
        _res = None
        _span = None

        string = ' ' + string + ' '
        _search_flag = pat.search(string)

        if _search_flag:
            _res_flag = True
            # _res = _search_flag.group(1)
            _res = str(_search_flag.group(1))
            _span = (_search_flag.span()[0], _search_flag.span()[1] - 2)

        return _res, _span, _res_flag, False


    def searchFunc(self, example):
        '''
        分批进行搜索
        '''

        raw_entity = example.get("entities", [])
        string = example.text

        if string.strip() == '':
            return raw_entity

        pointer_set = set()

        def _sub_search(sub_name, sub_func):
            pointer = 0
            _curr_string = deepcopy(string)
            while _curr_string:
                _search_res, _search_span_tuple, _search_flag, _search_span = sub_func(_curr_string)

                if _search_flag:

                    curr_pointer = _search_span_tuple
                    curr_pointer = (curr_pointer[0] + pointer, curr_pointer[1] + pointer)
                    _start_pointer = (curr_pointer[0] + 1) if _search_span else curr_pointer[0]
                    # _end_pointer = curr_pointer[1] - 2
                    _end_pointer = curr_pointer[1]
                    if curr_pointer in pointer_set:
                        pass

                    else:
                        _value = _search_res.upper() if sub_name == 'licePlat' else _search_res

                        # if sub_name == 'phoneNum':
                        #     _start_pointer -= 1
                        # if sub_name == "caseNum":
                        #     _end_pointer += 1
                        # if sub_name == "licePlat":
                        #     _start_pointer -= 1
                        #     _end_pointer += 1

                        pointer_set.add(curr_pointer)
                        raw_entity.append({'start': _start_pointer,
                                           'end': _end_pointer,
                                           'value': _value,
                                           'entity': self.component_config.get(sub_name, sub_name),
                                           "confidence": 1.})

                    pointer += curr_pointer[1]
                    _curr_string = _curr_string[pointer:]


                else:
                    break


        for _s_name, _s_func in zip(['idCard', 'phoneNum', 'caseNum', 'licePlat', 'bankNum'],
                                    [self._handle_id_card, self._handle_phone_number, self._handle_case_number,
                                     self._handle_licese_plat, self._handel_bank_number]):
            _sub_search(_s_name, _s_func)


        return raw_entity






# a = NumberPatternExtractor()
# target_list = a.searchFunc("我的手机号是吧那个是13684110597啊啊是13384615442的你说川BU512U是身份证号513902199511246711对的就是浙A05872这个")
# print(target_list)
