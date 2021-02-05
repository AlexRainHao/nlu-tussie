"""
method for constructing
    * tokens vocab
    * tags vocab
    * tagging schema handler
"""

from itertools import chain
from typing import Dict, Any, Text, List

__all__ = ["Vocab", "Tags", "TagScheme"]


class Vocab:
    """
    create vocab for tokens

    1. initial tokens except `UNK` could set with continuous index,
        E.X. init_tokens = {"SEP": 2, "CONJ": 3}, but
                init_tokens = {"SEP":2, "CONJ": 4} or init_tokens = {"SEP": 1, "CONJ": 2}
            would be illegal,

    2. filter token for constructed vocab by token corresponding frequency

    """
    # TODO: add load function
    def __init__(self, init_tokens = None):
        """

        Parameters
        ----------
        init_tokens : Dict
        """
        self.st = 0
        self.word2idx = {}
        self.idx2word = {}
        self.default_tokens = {}

        self.init_default_tokens(**(init_tokens or {}))

        self.word2count = {}

    @property
    def UNK(self):
        return 0

    @staticmethod
    def assert_continuous_init_tokens(default_tokens):
        """
        check whether continuous for default_tokens and initial tokens index,
        either overlapping index or non-continuous raise a error

        Parameters
        ----------
        default_tokens : Dict
        """
        default_tokens = sorted(default_tokens.items(), key = lambda ele: ele[1])

        st = default_tokens[0][1]

        if st < 0:
            raise ValueError("token id must greater than zero")

        for id, (tok, pot) in enumerate(default_tokens):
            if pot != st + id:
                raise ValueError("initial tokens are not continuous")

    @staticmethod
    def reverse_dict(dictionary):
        """reverse dictionary"""

        return {val: key for key, val in dictionary.items()}

    def init_default_tokens(self, **kwargs):
        """
        construct default vocab, means default tokens and initial tokens merged

        Parameters
        ----------
        init_tokens : Dict
        """

        _default_tokens = {"UNK": self.UNK}
        for tok, val in kwargs.items():
            _default_tokens[tok.upper()] = val

        self.assert_continuous_init_tokens(_default_tokens)
        self.default_tokens = _default_tokens

        self.word2idx.update(_default_tokens)
        self.idx2word.update(self.reverse_dict(self.word2idx))

    def add_token(self, token):
        """add new token to vocab

        Parameters
        ----------
        token : Text
        """
        self.st = self.__getattribute__("st") or max(self.word2idx.values())

        if token not in self.word2idx:
            self.word2idx[token] = self.st + 1
            self.idx2word[self.st + 1] = token
            self.word2count[token] = 1
            self.st += 1

        else:
            self.word2count[token] += 1

    def add_sentence(self, sentence):
        """add sentence to vocab

        Parameters
        ----------
        sentence : List[Text]
        """
        [self.add_token(token) for token in sentence]

    def filter_by_count(self, threshold = 2):
        """filter vocab by given frequency threshold,
        all the `word2idx`, `idx2word` and `word2count` would update
        """

        remain_tokens = list(filter(lambda x: self.word2count[x] > threshold, self.word2count))
        self.word2count = {token: self.word2count[token] for token in remain_tokens}

        token2idx = self.default_tokens.copy()
        st = max(token2idx.values())

        token2idx.update(dict(zip(remain_tokens, range(st + 1, st + 1 + len(remain_tokens)))))
        self.word2idx = token2idx
        self.idx2word = self.reverse_dict(self.word2idx)

        self.st = st

    def get_token_id(self, token):
        """get token id from vocab by given token

        Parameters
        ----------
        token : Text
        """
        return self.word2idx.get(token, self.UNK)

    def get_sequence_id(self, sequence):
        """get each token id from vocab for given sequence

        Parameters
        ----------
        sequence : List[Text]
        """

        return [self.get_token_id(tok) for tok in sequence]

    def get_token(self, idx):
        """get token from vocab by given index"""
        return self.idx2word.get(idx, self.idx2word.get(self.UNK))

    def get_sequence(self, idxs):
        """git each token from vocab for given sequence indexes

        Parameters
        ----------
        idxs : List[Int]
        """
        return [self.get_token(idx) for idx in idxs]

    def size(self):
        """vocab size"""
        return self.__len__()

    def __len__(self):
        return len(self.word2idx)

    def __repr__(self):
        return f"token vocab length {self.__len__()}"


class Tags:
    """
    create vocab for labels

    1. default token includes `START`, `STOP` set to fixed 0 and 1 separately,
        but different names allows
        E.X.
            tags = Tags()
            tags.START = "[CLS]"
            tags.STOP = "[SEP]"

    2. the initial tags index received also need to be continuous

    """

    def __init__(self, init_tags = None):
        """

        Parameters
        ----------
        init_tags : Dict
        """
        self.START = "<START>"
        self.STOP = "<END>"

        self.tag2idx = {}
        self.idx2tag = {}

        self.default_tags(**(init_tags or {}))

    def stopIdx(self):
        return self.get_tag_id(self.STOP, False)

    def startIdx(self):
        return self.get_tag_id(self.START, False)

    @property
    def START(self):
        return self._start

    @property
    def STOP(self):
        return self._stop

    @START.setter
    def START(self, start_tok):
        try:
            st_val = self.tag2idx.pop(self.START)

            self._start = start_tok

            self.tag2idx[start_tok] = st_val
            self.idx2tag[st_val] = start_tok
        except:
            self._start = start_tok

    @STOP.setter
    def STOP(self, stop_tok):
        try:
            ed_val = self.tag2idx.pop(self.STOP)

            self._stop = stop_tok

            self.tag2idx[stop_tok] = ed_val
            self.idx2tag[ed_val] = stop_tok
        except:
            self._stop = stop_tok

    @staticmethod
    def assert_continuous_init_tags(default_tags):
        """
        check whether continuous for default tags and initial tags index,
        either overlapping index or non-continuous raise a error

        Parameters
        ----------
        default_tags : Dict
        """
        default_tags = sorted(default_tags.items(), key = lambda ele: ele[1])

        st = default_tags[0][1]

        for id, (tag, pot) in enumerate(default_tags):
            if pot != st + id:
                raise ValueError("initial tags are not continuous")

    @staticmethod
    def reverse_dict(dictionary):
        """reverse dictionary"""

        return {val: key for key, val in dictionary.items()}

    def default_tags(self, **kwargs):
        """
        construct default tag vocab

        Parameters
        ----------
        kwargs: **Dict

        """

        _default_tags = {self.START: 0,
                         self.STOP: 1}

        for tag, val in kwargs.items():
            _default_tags[tag.upper()] = val

        self.assert_continuous_init_tags(_default_tags)

        self.tag2idx.update(_default_tags)
        self.idx2tag.update(self.reverse_dict(self.tag2idx))

    def add_tag(self, tag):
        """add new tag to vocab

        Parameters
        ----------
        tag : Text
        """

        if tag not in self.tag2idx:
            n_id = len(self.tag2idx)
            self.tag2idx[tag] = n_id
            self.idx2tag[n_id] = tag

    def add_taguence(self, taguence):
        """add taguence to vocab

        Parameters
        ----------
        taguence : List[Text]
        """
        [self.add_tag(tag) for tag in taguence]

    def get_tag_id(self, tag, update = False):
        """get tag id from vocab by given tag,
        arg or update determines whether add a new tag to vocab if
        given tag not shown in constructed vocab

        Parameters
        ----------
        tag : Text
        """

        if tag not in self.tag2idx:
            if update:
                n_id = len(self.tag2idx)
                self.tag2idx[tag] = n_id
                self.idx2tag[n_id] = tag
                return n_id

            else:
                raise ValueError("not found tag in given tag vocab")

        else:
            return self.tag2idx[tag]

    def get_taguence_id(self, taguence, update = False):
        """get each tag id for a taguence

        Parameters
        ----------
        taguence : List[Text]
        """

        return [self.get_tag_id(tag, update) for tag in taguence]
    
    def get_tag(self, idx):
        """get tag from vocab by given index

        Parameters
        ----------
        idx : Int
        """

        return self.idx2tag.get(idx, self.STOP)
        # return self.idx2tag.get(idx, "O")

    def get_taguence(self, idxs):
        """get each tag for a taguence

        Parameters
        ----------
        idxs : List[Int]
        """
        try:
            idxs = list(chain.from_iterable(idxs))
        except:
            pass
        return [self.get_tag(idx) for idx in idxs]

    def size(self):
        """tag vocab size"""
        return self.__len__()

    def __len__(self):
        return len(self.tag2idx)

    def __repr__(self):
        return f"tag vocab length {self.__len__()}"

class TagScheme:
    """a series of methods for tagging schema,
    E.X.
        whether tagging data satisfy `BIO` or `BIOES`
        convert `BIO` tagging schema to `BIOES` or vice versa
    """

    @classmethod
    def is_iob(cls, tags, schema = "BIO"):
        """whether a taguence satisfy `BIO`

        Parameters
        ----------
        tags : Optional[List[List[Text]], List[Text]]
        """
        if not tags:
            return False

        if isinstance(tags, list) and isinstance(tags[0], str):
            tags = [tags]

        uniq_tags = set(chain.from_iterable(tags))


        schema_rule = list("BIO") if schema.upper() == "BIO" else list("BOES")
        illegal_postfix = set(filter(lambda x: x.split('-')[0] not in schema_rule and x.split('-')[0] != "I", uniq_tags))

        if illegal_postfix:
            return False

        return True

    @classmethod
    def convert_iob1_to_iob2(cls, tags):
        """convert IOB1 to IOB2 for single taguence

        Parameters
        ----------
        tags : List[Text]
        """
        for i, tag in enumerate(tags):
            if tag == "O":
                continue

            prfix, *pofix = tag.split('-')

            if prfix == "B":
                continue

            elif i == 0 or tags[i-1] == "O": # O-I / ^I-O / ^I-B
                tags[i] = "B-" + '-'.join(pofix)

            elif tags[i - 1][1:] == tag[1:]: # B-I, I-I
                continue

            else: # I-
                tags[i] = "B-" + '-'.join(pofix)

        return tags

    @classmethod
    def convert_iob_to_iobes(cls, tags):
        """convert IOB2 to IOBES for single taguence

        Parameters
        ----------
        tags : List[Text]
        """

        tags += ["O"]
        bigram_tag = list(zip(*[tags[i:] for i in range(2)]))

        for i, (f_tag, b_tag) in enumerate(bigram_tag):
            if f_tag == "O":
                continue

            prfix, *pofix = f_tag.split('-')

            if f_tag.startswith("B") and not b_tag.startswith("I"): # for S
                tags[i] = "S-" + '-'.join(pofix)

            elif f_tag.startswith("I") and not b_tag.startswith("I"): # for E
                tags[i] = "E-" + '-'.join(pofix)

        return tags[:-1]

    @classmethod
    def convert_iobes_to_iob(cls, tags):
        """convert IOBES to IOB2 for single taguence

        Parameters
        ----------
        tags : List[Text]
        """
        for i, tag in enumerate(tags):
            if tag.startswith("S"):
                tags[i] = "B" + tag[1:]

            elif tag.startswith("E"):
                tags[i] = "I" + tag[1:]

        return tags

    @classmethod
    def run(cls, tags, name):
        """a simple pipeline wrapper"""
        _is_bio = cls.is_iob(tags, "BIO")
        _is_bioes = cls.is_iob(tags, "BIOES")

        if not _is_bio or not _is_bioes:
            raise ValueError("not supported `Non-BIO` or `Non-BIOES`scheme")

        if name == "BIO":
            if _is_bio:
                return cls.convert_iob1_to_iob2(tags)
            else:
                return cls.convert_iobes_to_iob(tags)

        elif name == "BIOES":
            if not _is_bio:
                tags = cls.convert_iob1_to_iob2(tags)
            return cls.convert_iob_to_iobes(tags)

