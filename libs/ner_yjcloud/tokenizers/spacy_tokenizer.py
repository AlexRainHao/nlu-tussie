'''
Usage: `spacy` tokenizer and self-defined entity ruler

support `spacy` tokenizer for chinese corpus
The model used pre-trained by `Spacy. Inc` named by `zh_core_web_lg`
We insist the view that use `pkuseg`, instead of `jieba`
as Tokenizer because of much of wrong word pos,

The two extra actions added:
    1. hot-word dictionary
        * the hot-word should saved as text(temporarily) in which row as the format of `yyh PERSON\n`
        * the hot-word would load in order of word length, therefore the word abcd would as token when
          both abc and abcd saved in dictionary simultaneously
        * the spacy engine would firstly add new tokens from 1st column in dict,
          and then add new entityRuler to used spacy model
        * All the hot-word config set from `spacy_utils.py`

    2. the pos each of word predicted by default model should restored during pipeline
        and could used in down-stream ner, Thus two character for NLU each pipeline named "pos" and "preEntity"
        added

For the tokenizer, the Spacy model seems not extract special token like space and it's pos,
So the tokenizer results need an alignment

The entities extracted by `Spacy` would restored during pipeline in convince,
but the string of entity may not intokenizer results,
for example, the entity of `fine day` may restored in tokenizer as `[fine, day]`.
So another alignment needed
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
import os
from typing import Any, List, Tuple
import sys
from collections import defaultdict

# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from ner_yjcloud.components import Component
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.tokenizers import Tokenizer, Token, SegPos, PreEntity
from ner_yjcloud.training_data import Message
from ner_yjcloud.training_data import TrainingData


SPACE_FLAG = "SPACE"

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    provides = ["tokens", "pos", "preEntity"]

    requires = ["spacy_doc", "spacy_nlp"]

    defaults = {'beam_config': {'beam_width': 4,
                                'beam_density': .0001}}

    def __init__(self, component_config = None):
        super(SpacyTokenizer, self).__init__(component_config)

        self.beam_config = self.component_config.get('beam_config')


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            token_nlp = example.get('spacy_nlp')
            token_res = self.tokenize(example.get('spacy_doc'), token_nlp)
            example.set("tokens", token_res[0])
            example.set('pos', token_res[1])
            example.set('preEntity', token_res[2])


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        token_nlp = message.get('spacy_nlp')
        token_res = self.tokenize(message.get('spacy_doc'), token_nlp)
        message.set("tokens", token_res[0])
        message.set('pos', token_res[1])
        message.set('preEntity', token_res[2])


    def tokenize(self, doc, nlp):
        # tokens = [Token(t.text, t.idx) for t in doc]
        # pos = SegPos([(t.text, t.pos_) for t in doc])
        entities = self.get_token_confidence(nlp, doc.text, doc.ents, self.beam_config)

        tokens, pos = self.add_space_token_to_res(doc)
        tokens, pos, entities = self.remove_null_tokens(tokens, pos, entities)

        return tokens, pos, entities


    @staticmethod
    def add_space_token_to_res(doc):
        tokens = [Token(t.text, t.idx) for t in doc]
        pos = [(t.text, t.pos_) for t in doc]
        # entities = [(t.text, t.start_char, t.end_char, t.label_) for t in doc.ents]

        assert len(tokens) == len(pos), 'have no same length of tokens and pos'
        # assert len(tokens) == len(pre_ent), 'have no same length of tokens and pre_ent'

        length = len(tokens)

        if len(doc.text) == 0:
            return tokens, pos

        if length == 0 and len(doc.text) != 0:
            return [Token(doc.text, 0)], [(doc.text, SPACE_FLAG)]

        if length == 1:
            if len(tokens[0].text) == len(doc.text):
                return tokens, [(t.text, t.pos_) for t in doc]

            else:
                new_tokens = [Token(' ' * (tokens[0].offset - 0), 0)] + \
                             tokens + \
                             [Token(' ' * (len(doc.text) - tokens[0].end), tokens[0].end)]

                new_pos = [(' ' * (tokens[0].offset - 0), SPACE_FLAG)] + \
                          [(t.text, t.pos_) for t in doc] + \
                          + [(' ' * (len(doc.text) - tokens[0].end), SPACE_FLAG)]

                return new_tokens, new_pos


        new_tokens = []
        new_pos = []

        last_start = 0

        while tokens:
            last_token = tokens.pop(0)
            last_pos = pos.pop(0)
            if last_token.offset > last_start:
                new_tokens.append(Token(' ' * (last_token.offset - last_start), last_start))
                new_pos.append((' ' * (last_token.offset - last_start), SPACE_FLAG))

            new_tokens.append(last_token)
            new_pos.append(last_pos)

            last_start = last_token.end

        if last_start < len(doc.text):
            new_tokens.append(Token(' ' * (len(doc.text) - last_start), last_start))
            new_pos.append((' ' * (len(doc.text) - last_start), SPACE_FLAG))

        return new_tokens, new_pos

    @staticmethod
    def remove_null_tokens(tokens, pos, entities):
        assert len(tokens) == len(pos), 'have no same length of tokens and pos'

        if not tokens:
            return tokens, pos, entities

        entity_dict = {x[1]:(x[2], x[0], x[3], x[4]) for x in entities}

        new_tokens, new_pos, new_entities = [], [], []
        offset_string = None

        for _token, _pos in zip(tokens, pos):
            if _token.text == '':
                continue

            # alignment for entity to tokenizer
            if offset_string:
                _token = Token(offset_string[0].text + _token.text, offset_string[0].offset)
                _pos = (offset_string[0].text + _pos[0], _pos[1])


            if entity_dict.get(_token.offset):
                if entity_dict.get(_token.offset)[0] == _token.end:
                    new_tokens.append(_token)
                    new_pos.append(_pos)
                    new_entities.append(PreEntity(_token.offset,
                                                  _token.end,
                                                  _token.text,
                                                  entity_dict.get(_token.offset)[2],
                                                  entity_dict.get(_token.offset)[3]))

                    offset_string = None

                else:
                    offset_string = (_token, _pos)

            else:
                new_tokens.append(_token)
                new_pos.append(_pos)
                new_entities.append(PreEntity())


        return new_tokens, SegPos(new_pos), new_entities


    @staticmethod
    def get_token_confidence(nlp, text, extracted_entities, beam_config):
        '''
        would split tokens from text again without standard NER pipeline

        Returns:

        '''

        doc = nlp.make_doc(text)
        for name, proc in nlp.pipeline:
            if name == "ner":
                continue
            else:
                doc = proc(doc)

        # idx2token_dict = {}
        # for tok in doc:
        #     idx2token_dict[tok.i] = [tok.idx, None, 0.]

        beams = nlp.entity.beam_parse([doc],
                                      beam_width = int(beam_config.get('beam_width', 4)),
                                      beam_density = float(beam_config.get('beam_density', .0001)))

        entity_scores = defaultdict(float)
        for beam in beams:
            for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score

        # entities = [(t.text, t.start_char, t.end_char, t.label_) for t in doc.ents]

        entities_dict = {(t.start, t.end):[t.text, t.label_, 0., t.start_char, t.end_char] for t in extracted_entities}

        for key in entity_scores:
            start, end, label = key
            if entities_dict.get((start, end)):
                if entities_dict[(start, end)][2] >= entity_scores.get((start, end, label)):
                    continue
                else:
                    entities_dict[(start, end)][1] = label
                    entities_dict[(start, end)][2] = entity_scores.get((start, end, label))

        pre_entities = [(val[0], val[3], val[4], val[1], val[2]) for _, val in entities_dict.items()]

        return pre_entities
