from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object


class Tokenizer(object):
    pass


class Token(object):
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)


class SegPos(object):
    '''
    Useg: POS obatined from tokenizer

    save tokens and pos list for some customize extractors
    used, for examples, nerDate, nerLoc et al.
    it's format different from Class Token show above.

    Notice that the values saved shouldn't edit because of
    all the extractors would use these value alone
    '''
    def __init__(self, tokens_res, entities = None):
        '''
        :param tokens_res: list(pair)
        '''
        self.entities = entities
        self.data = {}
        self.set(tokens = tokens_res)

    def set(self, tokens):
        '''
        :param tokens: list(pair)
        '''

        tokens_word, tokens_pos = [], []

        for word, flag in tokens:
            tokens_word.append(word)
            tokens_pos.append(flag)

        self.data.setdefault('word', tokens_word)
        self.data.setdefault('pos', tokens_pos)

    def get(self, prob, default = None):
        '''
        :param prob: ['word', 'pos']
        '''
        return self.data.get(prob, default)
        # pass

class PreEntity(object):
    '''
    pass
    '''
    def __init__(self,
                 start_id = None,
                 end_id = None,
                 text = None,
                 entity = None,
                 confidence = 0.):

        self.start_id = start_id
        self.end_id = end_id
        self.text = text
        self.entity = entity
        self.confidence = confidence

    def set(self):
        pass

    def get(self):
        pass
