# -*- coding: utf-8 -*-
'''
[depreciated]
This method has depreciated and `lac`/`spacy` or others method recommended

Usage: TIME, ORG, PERSON NER based on token POS and regular patterns

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
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

from ner_yjcloud.extractors import EntityExtractor
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message
from ner_yjcloud.training_data import TrainingData


class Stack():
    '''
    Queen actually
    '''
    def __init__(self, init):
        self.res = init

    def push(self, x):
        self.res.extend(x)

    def pop(self):
        if self.res:
            self.res.pop()
        else:
            pass


class preRegexDate:
    date_candi = ["今天", "明天", "后天", "昨天", "前天", "大前天", "大后天", "当天",
                  "这个月", "下个月", "下下个月", "上个月", "上上个月", "同月",
                  "下月", "上月", "月初", "月末", "月中", "月中旬",
                  "今年", "明年", "后年", "去年", "前年", "大前年", "大后年", "同年",
                  "上午", "下午", "中午", "早上", "早晨", "晚上", "傍晚", "当晚",
                  "点钟", "小时", "分钟", "刻",
                  "周一", "周二", "周三", "周四", "周五", "周六", "周末", "周日",
                  "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天", "星期日",
                  "礼拜一", "礼拜二", "礼拜三", "礼拜四", "礼拜五", "礼拜六", "礼拜天", "礼拜日",
                  "时期", "时刻"]

    time_psg_candi = ["m", "t", "mq"]

    joint_psg_candi = ['u', 'uj', 'c', 'e', 'y']

    pron_candi = ['这个', '那个']

    wrong_psg_candi = ['下来','前世','今生','后世']

    # name_psg_candi = ['nr', 'nt', 'ns', 'nz', 't']
    name_psg_candi = ['nr', 'nt', 't', 'nrt']

    configComp = {"date_dsr_candi": date_candi,
                  "date_psg_candi": time_psg_candi,
                  "joint_psg_candi": joint_psg_candi,
                  "stam_dsr_candi": pron_candi,
                  "wrong_psg_candi": wrong_psg_candi,
                  "name_psg_candi": name_psg_candi,
                  "tagName": "custDate",
                  "otherTag": {"nr": "PER",
                               "nt": "ORG"}}



class DatePsegExtractor(EntityExtractor):
    name = "date_pseg_extractor"

    provides = ["entities"]

    requires = ['pos']

    defaults = preRegexDate.configComp

    def __init__(self, component_config = None):
        # type: (Optional[Dict[Text, Text]]) -> None

        raise NotImplementedError("This method has depreciated and `lac`/`spacy` or others method recommended")

        super(DatePsegExtractor, self).__init__(component_config)
        self.component_config = component_config
        # self.tagName = "custDate"
        # self.otherTag = {'nr': "PER",
        #                  "nt": "ORG"}


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.component_config = config.for_component(self.name, self.defaults)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.timeExtract(message))

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
    def _remove_regex_pat(word_list, tag_list, pos_list):
        '''
        主要针对 xxx号
        '''
        pattern_1 = re.compile("[0-9零百十千一二三四五六七八九]{3,}[号|条]")
        pattern_2 = re.compile("[点\.分秒时年][0-9零一二三四五六七八九十]?[号|条]")
        pattern_3 = re.compile("[^0-9零百十千一二三四五六七八九][3三]十[0-1]?[号|条]")
        pattern_4 = re.compile("[0-9零百十千一二三四五六七八九]号楼")
        pattern_5 = '亩条元斤里米克块份幢多辆张把幢双支瓶左右台万岁'
        pattern_6 = re.compile(("号[0-9零一二三四五六七八九十百千万亿]{4,20}"))

        for i in range(len(word_list)):
            if not tag_list[i]:
                continue

            else:
                # _flag = True
                _str = ''.join(word_list[i])

                if pattern_3.search(_str):
                    continue

                if pattern_1.search(_str) or pattern_2.search(_str) or pattern_4.search(_str) or pattern_6.search(_str):
                    tag_list[i] = False
                    # _flag = False

                if sum(map(lambda x: x == 'm' or x == 't', pos_list[i])) == len(pos_list[i]):
                    if list(filter(lambda y: y[-1] in pattern_5, word_list[i])):
                        tag_list[i] = False

                    if ''.join(word_list[i]).replace('.', '').strip() == '':
                        tag_list[i] = False
                        # _flag = False

                # tag_list[i] = _flag


        return word_list, tag_list


    @staticmethod
    def _joint_word_list(head_list, curr_list, pattern):
        if not head_list:
            return True, True

        _flag = False
        curr_str = head_list[-1] + curr_list[0]

        if pattern.match(curr_str):
            _flag = True

        return _flag, False


    @staticmethod
    def _remove_nomean_word(curr_list, curr_pos, pattern, is_use=True):
        if not is_use:
            return curr_list, curr_pos, [], [], False

        if len(curr_pos) == 1 and 't' not in set(curr_pos):
            return curr_list, curr_pos, [], [], False

        _flag = True
        no_mean_list = []
        no_mean_pos = []

        if curr_pos[-1] not in pattern:
            return curr_list, curr_pos, [], [], True

        else:
            while _flag and curr_list:
                last_word = curr_list.pop()
                last_pos = curr_pos.pop()

                if last_pos in pattern:
                    no_mean_list.insert(0, last_word)
                    no_mean_pos.insert(0, last_pos)

                else:
                    _flag = False
                    curr_list.append(last_word)
                    curr_pos.append(last_pos)
            return curr_list, curr_pos, no_mean_list, no_mean_pos, True


    @staticmethod
    def _arrange_psg(result_nametuple_list, target_list, target_psg,
                     other_tag_psg, time_psg = 't'
                     ):

        if isinstance(target_psg, list):
            pass
        else:
            try:
                target_psg = list(target_psg)
            except:
                raise ValueError("Error in handling target_psg")

        target_psg.append(time_psg)

        if result_nametuple_list == []:
            return

        pointer = 0

        # pat = re.compile('^(\d+)$')
        pat = re.compile('^([0-9一二三四五六七八九十百千万亿岁余第条元]+)$')

        for seg in result_nametuple_list:
            seg_len = len(seg.segment)
            if seg.pos in target_psg or seg.is_date:
                if seg.pos == time_psg and not seg.is_date:
                    continue

                if seg.pos == "nr":
                    seg = seg._replace(pos = other_tag_psg.get("nr", "nr"))

                if seg.pos == "nt" or seg.pos == "nrt":
                    seg = seg._replace(pos = other_tag_psg.get("nt", "nt"))


                target_list.append({"start": pointer,
                                    "end": pointer + seg_len,
                                    "value": seg.segment,
                                    "entity": seg.pos})
                # 处理最后全为数字
                if pat.search(seg.segment):
                    target_list.pop(-1)

            pointer += seg_len


    def check_time_valid(self, word_list, pos_list, is_use_list, pattern):
        '''
        1. 末尾为无意义词，preRegexDate.joint_pag_candi
        2. sent[i] + sent[i+1] + ... 暂时考虑号、日、时、点、分、秒

        '''
        assert len(word_list) == len(pos_list)

        _nametuple = namedtuple('dateSegment', ['segment', 'is_date', 'pos'])
        res_word_list = []

        if len(word_list) <= 1:
            return [_nametuple(segment = ''.join(word_list[0]),
                               is_date = is_use_list[0],
                               pos = 't' if is_use_list[0] else 'n')]

        contion_patt = re.compile("^[年月天号日时点来多分秒0-9零一二三四五六七八九十]{1,2}[年月天号日时多来钟点分秒0-9零一二三四五六七八九十]{1,2}$")
        # contion_patt_2 = re.compile("([今明后昨前当]+天)?([上下中]+午)?(早[上晨]+)?([傍当]?晚上?)?[时点分秒0-9零一二三四五六七八九十]{1,2}$")
        curr_word = Stack(word_list.pop(0))
        curr_pos = Stack(pos_list.pop(0))

        curr_is_use = is_use_list.pop(0) # 第一个

        while word_list:
            head_word = word_list.pop(0)
            head_pos = pos_list.pop(0)
            head_is_use = is_use_list.pop(0) # 第二个

            # condition 2
            # print('-' *40)
            # print(curr_word.res)
            # print(head_word)

            _conti_flag, whether_space = self._joint_word_list(curr_word.res, head_word, contion_patt)
            _curr_is_use = deepcopy(curr_is_use)
            curr_is_use = head_is_use if whether_space else curr_is_use | head_is_use

            # print(curr_is_use)
            # print(head_is_use)

            if _conti_flag:

                curr_word.push(head_word)
                curr_pos.push(head_pos)
                curr_is_use = True


            #condition 1, save curr stack and init it

            else:
                _curr_list, _curr_pos, _no_mean_list, _no_mean_pos, is_date = self._remove_nomean_word(curr_word.res,
                                                                                                       curr_pos.res,
                                                                                                       pattern,
                                                                                                       is_use = curr_is_use)
                if _curr_list:
                    res_word_list.append(_nametuple(segment = ''.join(_curr_list),
                                                    is_date = is_date & _curr_is_use,
                                                    pos = self.component_config.get('tagName', 'custDate') \
                                                        if is_date&_curr_is_use \
                                                        else _curr_pos[-1]))

                if _no_mean_list:
                    res_word_list.append(_nametuple(segment = ''.join(_no_mean_list),
                                                    is_date = False,
                                                    pos = _no_mean_pos[-1]))

                # ====================

                curr_word = Stack(head_word)
                curr_pos = Stack(head_pos)
                curr_is_use = head_is_use

        if curr_word.res:
            res_word_list.append(_nametuple(segment = ''.join(curr_word.res),
                                            is_date = curr_is_use,
                                            pos = self.component_config.get('tagName', 'custDate') \
                                                if curr_is_use \
                                                else curr_pos.res[-1]))

        return res_word_list


    def timeExtract(self, example):

        raw_entity = example.get("entities", [])
        # raw_entity = []
        text = example.text
        tokenized_word = example.get('pos').get('word')
        tokenized_pos = example.get('pos').get('pos')
        # text = example

        if text.strip() == '':
            return raw_entity

        time_res = []
        pos_res = []
        is_date = []

        word = []
        pos = []
        keyDate = self.component_config['date_dsr_candi']
        # for k, v in psg.cut(text):
        for k,v in zip(tokenized_word, tokenized_pos):
            # print(k, v)
            if k in keyDate or v in self.component_config['date_psg_candi'] or \
                    v in self.component_config['joint_psg_candi']:
                if (not word and v in self.component_config['joint_psg_candi']) or \
                        k in self.component_config['wrong_psg_candi']:
                    time_res.append([k])
                    pos_res.append(['n'] * len(v))
                    is_date.append(False)
                else:
                    word.append(k)
                    pos.append(v)

            else:
                if word:
                    time_res.append(word)
                    pos_res.append(pos)
                    is_date.append(True)
                time_res.append([k])
                pos_res.append([v])
                is_date.append(False)
                word = []
                pos = []

        if word:
            time_res.append(word)
            pos_res.append(pos)
            is_date.append(True)

        # print(time_res)
        # print(pos_res)
        # print(is_date)

        time_res, is_date = self._remove_regex_pat(time_res, is_date, pos_res)

        result_collection_list = self.check_time_valid(time_res, pos_res, is_date,
                                                       self.component_config['joint_psg_candi'])

        self._arrange_psg(result_collection_list,
                          raw_entity,
                          self.component_config["name_psg_candi"],
                          self.component_config["otherTag"],
                          self.component_config["tagName"])

        # print(raw_entity)
        return raw_entity

