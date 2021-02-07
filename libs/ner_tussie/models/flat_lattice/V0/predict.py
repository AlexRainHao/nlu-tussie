import sys
import os.path as opt
import json

# PROLIBPATH = opt.abspath(opt.join(opt.abspath(__file__), *[opt.pardir] * 5))
# sys.path.append(PROLIBPATH)

# LIBPATH = opt.abspath(opt.join(opt.abspath(__file__), *[opt.pardir] * 2))
# sys.path.append(LIBPATH)
from typing import Dict, Union, List, Any

from time import time

from ner_tussie.models.flat_lattice.utils import get_bigrams, norm_static_embedding
from ner_tussie.models.flat_lattice.load_data import *
from ner_tussie.models.flat_lattice.V0.add_lattice import equip_chinese_ner_with_lexicon
from ner_tussie.models.flat_lattice.V0.utils_ import Trie

from ner_tussie.models.gpu_utils import getGPUs, getAvailabilityGPU
    
import torch

from fastNLP.io.model_io import ModelLoader


class objdict(object):
    def __init__(self, _dic):
        self.__dict__ = _dic

def set_config():
    """a set of path configuration"""
    
    cache_dir = "/home/admin/Flat_Lattice_NER.bak/cache"
    model_dir = "/home/admin/Flat_Lattice_NER.bak/out"

    return objdict({"cache_dir": cache_dir,
                    "model_dir": model_dir})


def load_device():
    """load a device which have largest GPU memory"""
    gpus = getGPUs()
    gpus = getAvailabilityGPU(gpus)
    
    return torch.device("cuda:%d" % gpus.id) if gpus != "cpu" else torch.device("cpu")
    # return torch.device("cuda:0") if gpus else torch.device("cpu")

def get_w_trie(w_list):
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    return w_trie
    
    
def load_vocab(model_dir, cache_dir, norm_embed = True, norm_lattic_embed = True):
    """load vocab, embedding, bi-gram, et.al"""
    # resume_ner_path = "/home/admin/Flat_Lattice_NER.bak/data/resume/toy"
    # raw_dataset_cache_name = "/home/admin/Flat_Lattice_NER.bak/cache/resume_trainClip:Truebgminfreq_1char_min_freq_1word_min_freq_1only_train_min_freqTruenumber_norm0load_dataset_seed100"
    # yangjie_rich_pretrain_unigram_path = '/home/admin/Flat_Lattice_NER.bak/pretrainedModel/gigaword_chn.all.a2b.uni.ite50.vec'
    # yangjie_rich_pretrain_bigram_path = '/home/admin/Flat_Lattice_NER.bak/pretrainedModel/gigaword_chn.all.a2b.bi.ite50.vec'
    # yangjie_rich_pretrain_word_path = '/home/admin/Flat_Lattice_NER.bak/pretrainedModel/ctb.50d.vec'
    #
    # datasets,vocabs,embeddings = load_resume_ner(resume_ner_path,
    #                                              yangjie_rich_pretrain_unigram_path,
    #                                              yangjie_rich_pretrain_bigram_path,
    #                                              _refresh=False,
    #                                              index_token=False,
    #                                              _cache_fp=raw_dataset_cache_name,
    #                                              char_min_freq=1,
    #                                              bigram_min_freq=1,
    #                                              only_train_min_freq=True)

    
    yangjie_rich_pretrain_char_and_word_path = f'{model_dir}/yangjie_word_char_mix.txt'
    
    w_list = load_yangjie_rich_pretrain_word_list(embedding_path = '',
                                                  _refresh=False,
                                                  _cache_fp=f"{cache_dir}/yj")
    cache_name = f"{cache_dir}/resume_lattice_only_train:False_trainClip:True_norm_num:0char_min_freq1bigram_min_freq1word_min_freq1only_train_min_freqTruenumber_norm0lexicon_yjload_dataset_seed100"
    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets = None, vocabs = None, embeddings = None,
                                                                  w_list = w_list, word_embedding_path = None,
                                                                  _refresh=False, _cache_fp=cache_name,
                                                                  only_lexicon_in_train=False,
                                                                  word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                                  number_normalized=0,
                                                                  lattice_min_freq=1,
                                                                  only_train_min_freq=True)
    
    if norm_embed:
        for k,v in embeddings.items():
            norm_static_embedding(v, norm_embed)
            
    if norm_lattic_embed:
        for k, v in embeddings.items():
            norm_static_embedding(v, norm_lattic_embed)
    
    return vocabs, embeddings, get_w_trie(w_list)


class InputExample():
    """wrapper function for obtained features feed into Flat Model"""
    def __init__(self,
                 sentence: str,
                 vocabs: Any,
                 embeddings: Any,
                 w_trie: Any,
                 device: Any,
                 ):
        self.sentence = sentence
        self.vocabs = vocabs
        self.embeddings = embeddings
        self.w_trie = w_trie
        self.device = device
        # self.get_w_trie()

        self.tokens = self.get_tokens(sentence)
        
        # define features
        self.seq_len_fea = self.get_seqlen_fea()
        
        self.char_fea = []
        self.bigram_fea = []
        self.lexicons_fea = []
        self.lattice_fea = []
        self.lex_len_fea = 0
        self.pos_s_fea = []
        self.pos_e_fea = []
        
    @staticmethod
    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result
        
    @staticmethod
    def concat(chars, lexicons):
        result = chars + list(map(lambda x:x[2],lexicons))
        return result
        
    @staticmethod
    def get_pos_s(lex_s, seq_len):
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    @staticmethod
    def get_pos_e(lex_e, seq_len):
        pos_e = list(range(seq_len)) + lex_e

        return pos_e
        
    def get_tokens(self, sentence):
        """convert a string sentence to a list consists of each character token"""
        return list(sentence)
        
    def get_bigrams(self, tokens):
        """convert uni-gram character tokens to bi-gram tokens"""
        return get_bigrams(tokens)
    
    def get_char_fea(self):
        """convert character to index feature"""
        for w in self.tokens:
            self.char_fea.append(self.vocabs["char"].to_index(w))
            
    def get_bigrams_fea(self):
        grams = self.get_bigrams(self.tokens)
        """convert bi-gram token to index feature"""
        for w in grams:
            self.bigram_fea.append(self.vocabs["bigram"].to_index(w))
    
    # ==========================
    # def get_w_trie(self):
    #     w_trie = Trie()
    #     for w in self.w_list:
    #         w_trie.insert(w)
    #
    #     self.w_trie = w_trie
        
    def get_lexicon(self):
        """convert character to lexicons feature"""
        lexicons = self.get_skip_path(self.tokens, w_trie = self.w_trie)
        
        self.lexicons_fea = lexicons
        self.lex_len_fea = self.get_lexlen_fea()
        
    def get_lattice_fea(self):
        # 1. get lexicons from chars feature
        # 2. get lattice from chars and lexicons
        lattice_fea = self.concat(self.char_fea, self.lexicons_fea)
        self.lattice_fea = [self.vocabs["lattice"].to_index(x) for x in lattice_fea]
    
    def get_seqlen_fea(self):
        """get input sentence length feature"""
        return len(self.sentence)
        # return 1
    
    def get_lexlen_fea(self):
        """get lexicons length feature"""
        return len(self.lexicons_fea)
        # return 1
    
    def get_pos_s_fea(self):
        self.pos_s_fea = self.get_pos_s(self.get_lex_s_fea(), self.seq_len_fea)
    
    def get_pos_e_fea(self):
        self.pos_e_fea = self.get_pos_e(self.get_lex_e_fea(), self.seq_len_fea)
    
    def get_lex_s_fea(self):
        return [x[0] for x in self.lexicons_fea]
    
    def get_lex_e_fea(self):
        return [x[1] for x in self.lexicons_fea]
    
    def __call__(self):
        self.get_char_fea()
        self.get_bigrams_fea()
        self.get_lexicon()
        self.get_pos_s_fea()
        self.get_pos_e_fea()
        self.get_lattice_fea()

        lattice_fea = torch.LongTensor(self.lattice_fea).view(1, -1).to(self.device)
        bigram_fea = torch.LongTensor(self.bigram_fea).view(1, -1).to(self.device)
        seq_len_fea = torch.LongTensor([self.seq_len_fea]).to(self.device)
        lex_len_fea = torch.LongTensor([self.lex_len_fea]).to(self.device)
        pos_s_fea = torch.LongTensor(self.pos_s_fea).view(1, -1).to(self.device)
        pos_e_fea = torch.LongTensor(self.pos_e_fea).view(1, -1).to(self.device)
        
        return [lattice_fea, bigram_fea, seq_len_fea,
                lex_len_fea, pos_s_fea, pos_e_fea]
    

def load_model(model_dir, device):
    """load trained model and its parameters, model name must as flat_ner.model"""
    
    model = ModelLoader.load_pytorch_model(model_dir + '/flat_ner.model')
    
    if 'cpu' in device.type:
        model.to("cpu")
    else:
        model.to("cuda")
        
    model.eval()
        
    return model
    

def wrapper_results(text, labels):
    """convert label prediction to expected format"""

    def init_entity(idx):
        return {'start': idx + 1, 'value': '', 'entity': None, 'end': 0, "confidence": 1.0}
    
    entities = []
    entity = {'start': 0, 'value': '', 'entity': None, 'end': 0, "confidence": 1.0}
    
    for idx, (chr, label) in enumerate(zip(text, labels)):
        if label == 'O' and entity.get('value', ''):
            entity["end"] = entity.get('start') + len(entity.get('value'))
            entities.append(entity)
            entity = init_entity(idx)
        
        elif label == 'O':
            entity = init_entity(idx)

        elif label.startswith("S") and label.split('-')[-1] in ["TITLE", "EDU"]:
            entity = init_entity(idx)

        elif label.startswith('S') or label.startswith('E'):
            entity["value"] += chr
            entity["entity"] = label.split('-')[-1]
            entity["end"] = entity["start"] + len(entity.get('value'))
            entities.append(entity)
            entity = init_entity(idx)

        else:
            entity["value"] += chr
            
    return entities

def predict(text, vocabs, embeddings, w_trie, device, model, dump_func = wrapper_results):
    """predition for given sentence and wrapper the result labels to target format"""
    if len(text) == 0:
        return []

    if len(text) > 273:
        text = text[:273]


    example = InputExample(text, vocabs, embeddings, w_trie, device)
    
    pred_res = model(*example(), [])
    pred_label = []
    
    for w in pred_res["pred"].tolist()[0]:
        pred_label.append(vocabs["label"].to_word(w))

    # dump to json format
    pred_label = dump_func(text, pred_label)
    return pred_label
    
    
if __name__ == "__main__":
    cache_dir = "./cache"
    model_dir = "./out"

    device = load_device()
    vocabs, embeddings, w_trie = load_vocab(model_dir, cache_dir)
    model = load_model(model_dir, device)

    predict(text = "2020年5月，李明生，本科学历，曾在上海新华社担任工程师一职",
            vocabs = vocabs, embeddings = embeddings, w_trie = w_trie, device = device,
            model = model,
            dump_func = wrapper_results)


