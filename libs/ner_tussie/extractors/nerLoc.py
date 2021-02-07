# -*- coding: utf-8 -*-
'''
[depreciated]
This method has depreciated and `lac`/`spacy` or others method recommended

Usage: LOC NER based on token POS and regular patterns

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
# import jieba.posseg as psg
from collections import namedtuple
from copy import deepcopy

from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_tussie.extractors import EntityExtractor
from ner_tussie.model import Metadata
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData

class Stack():
    '''
    Queen actually
    '''
    def __init__(self, init):
        if isinstance(init, list):
            self.res = init
        else:
            self.res = [init]

    def push(self, x):
        self.res.append(x)

    def pop(self):
        if self.res:
            self.res.pop(0)
        else:
            pass

class preRegexLoc:
    road_candi = "[街|大]?[路|道|巷|街]$" # 五常街道188号, regex + m/mq/eng
    resid_candi = [] # 西溪北苑, ns 靠数据吧
    build_candi = "[a-zA-Z0-9百十千万一二三四五六七八九零][号]?[幢|楼|栋|室|号|楼|巷|屯|乡|镇]$"
    no_build_candi = "(多少|几|一|马)(号)?[路|道|号|幢|楼|栋|室|楼|巷|屯|乡]$"
    unit_candi = "单元$"

    step_candi = ['m', 'mq', 'eng', 'u', 'uj', 'c', 'b', 'e','q'] # 窗口适用
    pron_candi = ['这个', '那个'] # 窗口适用， 避免我、他这些代词

    step_window = 2 # 窗口适用

    start_from_ns = True # 控制贪婪

    configComp = {"road_dsr_candi": road_candi,
                  "resid_dsr_candi": resid_candi,
                  "build_dsr_candi": build_candi,
                  "nobuild_dsr_candi": no_build_candi,
                  "unit_dsr_candi": unit_candi,
                  "window_dsr_step": step_window,
                  "step_dsr_candi": step_candi,
                  "pron_dsr_candi": pron_candi,
                  "start_from_ns": start_from_ns,
                  "tagName": "custLoc"}


class LocPsegExtractor(EntityExtractor):
    name = "loc_pseg_extractor"

    provides = ["entities"]

    requires = ['pos']

    defaults = preRegexLoc.configComp

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        raise NotImplementedError("This method has depreciated and `lac`/`spacy` or others method recommended")


        super(LocPsegExtractor, self).__init__(component_config)
        # self.tagName = "RESIDENT"
        self.component_config = component_config


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.locExtractor(message))

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
    def _handle_config(config_dict):
        config_dict["road_dsr_candi"] = re.compile(config_dict["road_dsr_candi"])
        config_dict["build_dsr_candi"] = re.compile(config_dict["build_dsr_candi"])
        config_dict["unit_dsr_candi"] = re.compile(config_dict["unit_dsr_candi"])
        config_dict["nobuild_dsr_candi"] = re.compile(config_dict["nobuild_dsr_candi"])

        return config_dict


    @staticmethod
    def _poseg_sentence(token_word, token_pos):
        seg_word = []
        seg_pos = []
        is_loc = []

        # for k, v in psg.cut(text):
        for k, v in zip(token_word, token_pos):
            seg_word.append(k)
            seg_pos.append(v)

            is_loc.append(True if v == 'ns' else False)

        return seg_word, seg_pos, is_loc


    @staticmethod
    def _joint_word_list(curr_stack, curr_pos_stack,
                         tail_word, tail_pos,
                         last_header = 1):
        '''
        :returns: None ---> 直接推出栈
                  0 ---> 延长栈
                  >0 ---> 窗口延长栈
        '''

        # ns词入窗
        if tail_pos == 'ns':
            curr_stack.push(tail_word)
            head_pointer = 0


        else:
            curr_stack.push(tail_word)
            curr_pos_stack.push(tail_pos)
            head_pointer = last_header if last_header else len(curr_stack.res) - 1

        return head_pointer


    @staticmethod
    def _regex_search(curr_stack, head_pointer, config):

        search_flag = False

        if head_pointer == 0:
            return True

        # elif not head_pointer:
        #     return False

        curr_word_list = curr_stack.res
        curr_word_str = ''.join(curr_word_list)

        for op in ['road_dsr_candi', 'build_dsr_candi', 'unit_dsr_candi']:
            _search_op = re.compile(config[op]).search(curr_word_str)
            _search_flag = True if _search_op else False

            search_flag = _search_flag or search_flag

            if search_flag:
                return search_flag

        return search_flag


    @staticmethod
    def _justify_search(curr_stack, curr_pos_stack, head_pointer, search_flag):
        '''
        如果搜到且0, 重置window step, --> HP-0
        如果搜到且非0, 迭代window step, --> HP-50
        没搜到， 从header point推出, --> HP-100

        [更新当前栈]
        '''
        no_mean_list = None
        no_mean_pos_list = None

        if search_flag:
            curr_stack.res = [''.join(curr_stack.res)]
            curr_pos_stack.res = ['ns']
            return_flag = 'HP-0'

        else:
            no_mean_list = curr_stack.res[head_pointer:]
            no_mean_pos_list = curr_pos_stack.res[head_pointer:]
            return_flag = 'HP-50'

        return return_flag, no_mean_list, no_mean_pos_list


    def check_loc_valid(self, word_list, pos_list, is_loc_list, pattern):
        '''
        1. 如果是ns开始搜索 *
        '''
        assert len(word_list) == len(pos_list)

        _nametuple = namedtuple('locSegment', ['segment', 'pos', 'is_loc'])
        res_example = []

        if len(word_list) <= 1:
            return [_nametuple(segment = ''.join(word_list),
                               is_loc = is_loc_list[0],
                               pos = pos_list[0])]

        windowSize = pattern['window_dsr_step']

        # 入栈
        curr_word_stack = Stack(word_list.pop(0))
        curr_pos_stack = Stack(pos_list.pop(0))

        curr_pointer = 0 # 窗口计数
        last_header = 0 # 控制窗口延长
        window_pos = None


        while word_list:

            #print('*' * 40)
            #print(curr_word_stack.res)
            #print(curr_pos_stack.res)

            # 待检测单项
            tail_word = word_list.pop(0)
            tail_pos = pos_list.pop(0)

            if not curr_word_stack.res[-1] or not curr_pos_stack.res[-1]:
                curr_word_stack.push(tail_word)
                curr_pos_stack.push(tail_pos)
                curr_word_stack.pop()
                curr_pos_stack.pop()
                continue

            # ==========================
            # 控制贪婪与否
            # 关闭这步可避免问题1
            if pattern['start_from_ns']:
                if curr_pos_stack.res[0] != 'ns' and curr_pos_stack.res[0]:
                    # res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                    #                               pos='n',
                    #                               is_loc=False))
                    for word in curr_word_stack.res:
                        res_example.append(_nametuple(segment = word, pos = 'n', is_loc = False))
                    curr_word_stack = Stack(init = tail_word)
                    curr_pos_stack = Stack(init = tail_pos)
                    continue
            # ==========================

            step_header = self._joint_word_list(curr_word_stack,
                                                curr_pos_stack,
                                                tail_word,
                                                tail_pos,
                                                last_header)
            step_search_flag = self._regex_search(curr_word_stack,
                                                  step_header,
                                                  self.component_config)
            step_return_flag, no_mean_list, no_mean_post_list = self._justify_search(curr_word_stack,
                                                                                     curr_pos_stack,
                                                                                     step_header,
                                                                                     step_search_flag,
                                                                                     )

            # ======================
            # 按 [返回码] 进行 栈处理
            if step_return_flag == 'HP-0':
                curr_pointer = 0
                last_header = 0
                window_pos = None
            elif step_return_flag == 'HP-50':
                if curr_pointer < windowSize:
                    curr_pointer += 1
                    last_header = step_header
                    window_pos = window_pos if window_pos else curr_pos_stack.res[0]

                else:
                    curr_word_stack.res = curr_word_stack.res[:step_header]
                    curr_pos_stack.res = curr_pos_stack.res[:step_header]
                    # res_example.append(_nametuple(segment = ''.join(curr_word_stack.res),
                    #                               pos = window_pos,
                    #                               is_loc = True if window_pos == 'ns' else False))


                    if window_pos:
                        res_example.append(_nametuple(segment = ''.join(curr_word_stack.res),
                                                      pos = window_pos,
                                                      is_loc = True))
                    else:
                        for word in curr_word_stack.res:
                            res_example.append(_nametuple(segment = word,
                                                          pos = window_pos,
                                                          is_loc = False))

                    if no_mean_list:
                        # curr_word_stack = Stack(init = ''.join(no_mean_list))
                        # curr_pos_stack = Stack(init = 'n')
                        curr_word_stack = Stack(init = no_mean_list)
                        curr_pos_stack = Stack(init = ['n'] * len(no_mean_list))
                    else:
                        curr_word_stack = Stack(init = '')
                        curr_pos_stack = deepcopy(curr_word_stack)
                    curr_pointer = 0
                    last_header = 0
                    window_pos = None

        # 如果curr_stack还有值,1)最后一个为ns 2)最后一个处于窗口迭代中 3)最后一个为tail_word
        if curr_word_stack.res and curr_pointer == 0 and curr_pos_stack.res[0] == 'ns':
            res_example.append(_nametuple(segment = ''.join(curr_word_stack.res),
                                          pos = self.component_config.get('tagName', 'custLoc'),
                                          is_loc = True))
        elif curr_word_stack.res and curr_pointer == 0 and tail_pos != 'ns':
            # res_example.append(_nametuple(segment = ''.join(curr_word_stack.res),
            #                               pos = 'n',
            #                               is_loc = False))
            for word in curr_word_stack.res:
                res_example.append(_nametuple(segment = ''.join(word),
                                              pos = 'n',
                                              is_loc = False))

        elif curr_word_stack.res and curr_pointer:
            curr_word_stack.res = curr_word_stack.res[:step_header]
            curr_pos_stack.res = curr_pos_stack.res[:step_header]
            res_example.append(_nametuple(segment = ''.join(curr_word_stack.res),
                                          pos = self.component_config.get('tagName', 'custLoc'),
                                          is_loc = True))
            # res_example.append(_nametuple(segment = ''.join(no_mean_list),
            #                               pos = 'n',
            #                               is_loc = False))
            for word in no_mean_list:
                res_example.append(_nametuple(segment = word,
                                              pos = 'n',
                                              is_loc = False))

        return res_example


    def auxilary_search(self, last_example, pattern):
        '''
        主要针对问题1，直接搜正则
        接收的为主搜索的结果
        '''

        windowSize = pattern['window_dsr_step']
        _nametuple = namedtuple('locSegment', ['segment', 'pos', 'is_loc'])
        res_example = []
        window_step = 0
        # header_pointer = 1

        def _search_func(string, config=pattern):
            _return_flag = False

            for pat in [config["road_dsr_candi"],
                        config["build_dsr_candi"],
                        config["unit_dsr_candi"]]:
                if re.compile(pat).search(string):
                    _return_flag = _return_flag or True

                # print(f'regex flag {_return_flag}')

            if re.compile(config["nobuild_dsr_candi"]).search(string):
                _return_flag = False
            # print(f'regex flag after {_return_flag}')

            patter_time = re.compile("[年月日分秒时]")

            if patter_time.findall(string):
                _return_flag = False

            return _return_flag

        def _whether_number(word):
            pattern = re.compile('[0-9零一二三四五六七八九十百幺]')

            return True if pattern.findall(word) else False



        if not last_example:
            return last_example

        if len(last_example) == 1:
            return last_example

        curr_segment = last_example.pop(0)
        curr_word_stack = Stack(init=curr_segment.segment)
        header_pointer = 0 if curr_segment.is_loc else 1

        while last_example:
            tail_segment = last_example.pop(0)
            tail_word = tail_segment.segment
            tail_pos = tail_segment.is_loc

            #print('*' * 40)
            #print(curr_word_stack.res)
            #print(tail_word)
            #print(tail_pos)
            #print(header_pointer)
            #print(window_step)

            if tail_pos:
                if curr_word_stack.res[0] and header_pointer == 0:
                    res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                                                  pos=self.component_config.get('tagName', 'custLoc'),
                                                  is_loc=True))
                elif curr_word_stack.res[0] and header_pointer != 0:
                    res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                                                  pos='n',
                                                  is_loc=False))
                res_example.append(tail_segment)
                header_pointer = 1
                window_step = 0
                curr_word_stack.res = ['']
                continue

            if curr_word_stack.res == ['']:
                curr_word_stack.res = [tail_word]
                header_pointer = 1
                window_step = 0
                continue

            curr_word_stack.push(tail_word)

            if _whether_number(tail_word):
                window_step += 0

            else:

                window_step += 1

            #print('-' * 40)
            #print(curr_word_stack.res)
            #print(window_step)

            if window_step < windowSize:
                #print('+' * 40)
                #print(curr_word_stack.res)
                #print(''.join(curr_word_stack.res))
                _search_flag = _search_func(''.join(curr_word_stack.res))
                # print(f'Search Flag {_search_flag}')
                if _search_flag:
                    curr_word_stack = Stack(''.join(curr_word_stack.res))
                    header_pointer = 0
                    window_step = 0
                else:
                    pass

            else:
                for i in range(windowSize):
                    _push_back_val = curr_word_stack.res.pop(0)
                    if header_pointer == 0:
                        _push_back_pos = self.component_config.get('tagName', 'custLoc')
                        _push_back_isloc = True
                    else:
                        _push_back_pos = 'n'
                        _push_back_isloc = False

                    res_example.append(_nametuple(segment=_push_back_val,
                                                  pos=_push_back_pos,
                                                  is_loc=_push_back_isloc))
                header_pointer = 1
                window_step = 0

        if curr_word_stack.res[0] and header_pointer == 0:

            if re.compile('[a-zA-Z0-9幺一二三四五六七八九零]{5,6}').search(''.join(curr_word_stack.res)):
                res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                                              pos = 'n',
                                              is_loc = False))
            else:
                res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                                              pos=self.component_config.get('tagName', 'custLoc'),
                                              is_loc=True))
        elif curr_word_stack.res[0] and header_pointer != 0:
            res_example.append(_nametuple(segment=''.join(curr_word_stack.res),
                                          pos='n',
                                          is_loc=False))

        return res_example


    def arange_res(self, res_example, target_list):
        '''
        连续的pos连起来
        '''
        if not res_example:
            return

        if len(res_example) == 1:
            seg = res_example.pop(0)
            if seg.is_loc:
                target_list.append({"start": 0,
                                    "end": len(seg.segment),
                                    "value":seg.segment,
                                    "entity": self.component_config.get('tagName', 'custLoc')})

            return target_list


        pointer = 0
        curr_str = ''

        for example in res_example:
            seg = example.segment
            pos = example.is_loc

            if not pos and not curr_str:
                pointer += len(seg)

            elif pos:
                curr_str += seg

            elif not pos and curr_str:
                target_list.append({"start": pointer,
                                    "end": pointer + len(curr_str),
                                    "value": curr_str,
                                    "entity": self.component_config.get('tagName', 'custLoc')})
                pointer += len(curr_str) + len(seg)
                curr_str = ''

        if curr_str:
            target_list.append({"start": pointer,
                                "end": pointer + len(curr_str),
                                "value": curr_str,
                                "entity": self.component_config.get('tagName', 'custLoc')})



    def locExtractor(self, example):
        raw_entity = example.get('entities', [])
        text = example.text

        if text.strip() == '':
            return raw_entity

        tokenized_word = example.get('pos').get('word')
        tokenized_pos = example.get('pos').get('pos')

        seg_word, seg_pos, is_loc = self._poseg_sentence(tokenized_word,
                                                         tokenized_pos)
        stack_1_res = self.check_loc_valid(seg_word, seg_pos, is_loc, self.component_config)
        stack_2_res = self.auxilary_search(stack_1_res, self.component_config)

        self.arange_res(stack_2_res, raw_entity)

        return raw_entity


