"""
Extractor based on
    * Embed_CRF
    * BiLSTM_CRF
    * Bert_CRF
    * Bert_BiLSTM_CRF

This pipeline component is flexible, where
    * all the raw text processing could be customized
    * `BIO`, `BIOES` tagging scheme accepts
    * initial index for token vocab, tag vocab accepts
    * segment features could feed into `embedding`  based model, but `bert` temporarily
    * post processing for predicted label could be customized
    * auto using GPU

"""

from typing import Any, Dict, List, Optional, Text, Tuple, Callable, Union

import logging
import os
import os.path as opt

from time import time
from collections import namedtuple
import random
from tqdm import tqdm
import pickle
from shutil import copyfile
import jieba
from multiprocessing import cpu_count
import numpy as np

from ner_tussie.extractors import EntityExtractor, tagNormalizer
from ner_tussie.model import Metadata
from ner_tussie.training_data import Message
from ner_tussie.training_data import TrainingData
from ner_tussie.utils import write_to_file
from ner_tussie.train import RasaNLUModelConfig
from ner_tussie.normalizer import Normalizers

from ner_tussie.models.gpu_utils import getGPUs, getAvailabilityGPU
from ner_tussie.models import nerEvalCounter, report_epoch_score_matr
from ner_tussie.models.bert.tokenization import FullTokenizer
from ner_tussie.models.bert.modeling import BertConfig

# =============
from ner_tussie.models.bilstm_jection.utils import *
from ner_tussie.models.bilstm_jection.vocab import Vocab, Tags, TagScheme
from ner_tussie.models.bilstm_jection.model import BiLSTM_CRF, BertBiLSTM_CRF
from ner_tussie.models.bert.modeling import get_assignment_map_from_checkpoint

import tensorflow as tf

logger = logging.getLogger(__name__)

BILSTM_CRF_MODEL_NAME = "bilstm_jection"

class Processor:
    """public template for extracting examples and features processor,
    and batch sampler method
    """
    
    def __init__(self):
        self.examples = []
        self.features, self.train_features, self.dev_features = [], [], []
        
        self.Vocab = None
        self.Tags = None
    
    def create_examples(self, *args, **kwargs):
        """pass"""
        raise NotImplementedError()
    
    def create_features(self, *args, **kwargs):
        raise NotImplementedError()
    
    def _convert_entity(self, example: Message):
        """get training entities for a example"""
        entities = example.get("entities", [])
        
        return [(ent["start"], ent["end"], ent["entity"]) for ent in entities]
    
    def _convert_exam(self, text: Text,
                      words: List[Text],
                      ent_offsets: List[Any],
                      tag_schema: Text,
                      normalizer: Union[Callable, List[Callable]],
                      lower: bool = False) -> Tuple[List, List, List]:
        """
        obtain each example text token, word, label,
        label would be converted to given tag schema

        Parameters
        ----------
        text: a string of text
        words: each word for text
        ent_offsets: tagged entities information
        tag_schema: expected tag schema outputs
        normalizer: raw text Normalizers pipeline
        lower: whether use lower case

        """
        if isinstance(normalizer, Callable):
            text = normalizer(text)
        else:
            for norm in normalizer:
                text = norm(text)
        
        text = list(text.lower() if lower else text)
        
        tags = ["O" for _ in range(len(text))]
        
        for spt, ept, ent in ent_offsets:
            tags[spt] = "B-" + ent
            for i in range(spt + 1, ept):
                tags[i] = "I-" + ent
        
        tags = TagScheme.run(tags, tag_schema.upper())
        
        return text, words, tags
    
    def train_size(self):
        return len(self.train_features)
    
    def dev_size(self):
        return len(self.dev_features)
    
    def size(self):
        return self.__len__()
    
    def __len__(self):
        return len(self.examples)
    
    def _padding(self, features: List):
        return features
    
    def batcher(self, features, batch_size, if_shuffle=True):
        """batch sampler, each batch would padded to same length"""
        features_cp = features.copy()
        if if_shuffle:
            random.shuffle(features_cp)
        
        size = len(features)
        
        for ndx in range(0, size, batch_size):
            batch_features = features_cp[ndx: min(ndx + batch_size, size)]
            
            yield self._padding(batch_features)

class VocabFeatureProcessor(Processor):
    """`embedding` method feature processor"""
    
    def __init__(self, tag_schema, filter_threshold, initial_tokens=None, initial_tags=None, lower_case = False):
        super(VocabFeatureProcessor, self).__init__()
        
        self.lower_case = lower_case
        
        self.tag_schema = tag_schema
        self.filter_threshold = filter_threshold
        self.Vocab = Vocab(init_tokens=initial_tokens)
        self.Tags = Tags(init_tags=initial_tags)
    
    def create_examples(self,
                        examples: List[Message],
                        normalizer: Union[Callable, List[Callable]]):
        """
        * construct token vocab
        * construct tag vocab
        * obtain each example wrapper by `namedtuple`
        """
        ds = namedtuple("dataset", ["text", "words", "tags"])
        
        for example in examples:
            tokens = example.get("tokens", [])
            
            if tokens:
                tokens = [x.text for x in tokens]
            
            else:
                tokens = jieba.lcut(example.text)
            
            entity_offsets = self._convert_entity(example)
            exam = self._convert_exam(example.text,
                                      tokens,
                                      entity_offsets,
                                      self.tag_schema,
                                      normalizer,
                                      self.lower_case)
            exam = ds(*exam)
            
            self.Vocab.add_sentence(exam.text)
            self.Tags.add_taguence(exam.tags)
            
            self.examples.append(exam)
        
        self.Vocab.filter_by_count(threshold=self.filter_threshold)
    
    def create_features(self, examples, is_training = True, use_seg=True):
        """obtain each training example features from constructed vocabs, including
            * token ids
            * segment ids
            * label ids
        """
        features = prepare_dataset(examples=examples,
                                   vocabs=self.Vocab,
                                   tags=self.Tags,
                                   use_seg=use_seg)
        
        if is_training:
            self.train_features += features
        else:
            self.dev_features += features
        
        self.features += features
    
    def _padding(self, feature):
        tok_ids = [x.token_ids for x in feature]
        seg_ids = [x.seg_ids for x in feature]
        tag_ids = [x.tag_ids for x in feature]
        
        max_len = max(len(x) for x in tok_ids)
        
        for i, (tok, seg, tag) in enumerate(zip(tok_ids, seg_ids, tag_ids)):
            pad_len = max_len - len(tok)
            tok_ids[i] += [self.Vocab.UNK] * pad_len
            seg_ids[i] += [0] * pad_len
            tag_ids[i] += [self.Tags.stopIdx()] * pad_len
        
        return (tok_ids, seg_ids, tag_ids)


class BertFeatureProcessor(Processor):
    """`bert` method feature processor"""
    
    def __init__(self, tag_schema, max_seq_len, vocab_file, bert_config, lower_case=False, initial_tags=None):
        super(BertFeatureProcessor, self).__init__()
        
        self.tag_schema = tag_schema
        self.initial_tags = initial_tags
        
        self.max_seq_len = max_seq_len
        
        self.lower = lower_case
        
        self.bert_config = bert_config
        
        self.tokenizer = self._load_bert_tokenizer(vocab_file,
                                                   max_seq_len,
                                                   lower_case)
        
        self.Tags = Tags(init_tags=initial_tags)
        
        self.idx = 0
    
    def _load_bert_tokenizer(self, vocab_file, max_seq_len, lower=False):
        """load bert tokenizer"""
        if not opt.exists(vocab_file):
            raise FileNotFoundError("bert vocab file not found")
        
        if not opt.exists(self.bert_config):
            raise FileNotFoundError("bert config file not found")
        
        self.bert_config = BertConfig.from_json_file(self.bert_config)
        
        max_position = self.bert_config.max_position_embeddings - 2
        
        if max_seq_len > max_position:
            logger.warning(f"max sequence length must less than bert required,"
                           f"training will continue where max_seq_len = {max_position}")
            
            self.max_seq_len = max_position
        
        tokenizer = FullTokenizer(vocab_file=vocab_file,
                                  do_lower_case=lower)
        
        return tokenizer
    
    def create_examples(self,
                        examples: List[Message],
                        normalizer: Union[Callable, List[Callable]]):
        """
        * construct tag vocab
        * obtain each example wrapper by `InputExample`
        """
        
        for example in examples:
            entity_offsets = self._convert_entity(example)
            
            tokens = []
            
            exam = self._convert_exam(example.text,
                                      tokens,
                                      entity_offsets,
                                      self.tag_schema,
                                      normalizer,
                                      self.lower)
            exam = InputExample(guid=str(self.idx),
                                text=exam[0],
                                label=exam[2])
            self.examples.append(exam)
            
            self.Tags.add_taguence(exam.label)
            
            self.idx += 1
    
    def create_features(self, examples: List, is_training = True):
        """obtain each training example feature, including
            * token ids
            * mask ids
            * type ids
            * label ids
        note that these feature as format of single sentence, but pairs of Bert,
        and all the examples features padding to a max length defined
        """
        features = prepare_bert_dataset(examples=examples,
                                        tokenizer=self.tokenizer,
                                        max_seq_len=self.max_seq_len,
                                        tags=self.Tags)
        if is_training:
            self.train_features += features
        else:
            self.dev_features += features
        
        self.features += features
    
    def _padding(self, feature: List[InputFeature]):
        input_ids = [x.input_ids for x in feature]
        input_mask = [x.input_mask for x in feature]
        input_type_ids = [x.input_type_ids for x in feature]
        label_ids = [x.label_ids for x in feature]
        
        return (input_ids, input_mask, input_type_ids, label_ids)


class BiLSTMCrfEntityExtractor(EntityExtractor):
    name = "ner_bilstm_crf"
    
    provides = ["entities"]
    
    requires = ["tokens"]
    
    defaults = {"tag_schema": "BIO",
                "embedding": "bert",
                "normalizers": {"digit_zero_flat": True},
                "filter_threshold": 2,
                "initial_tokens": {},
                "initial_tags": {},
    
                "lower_case": True,
                "use_seg": False,
                "token_dim": 100,
                "seg_dim": 20,
                "hidden_dim": 100,
    
                "crf_only": False,
                "bert_path": "/home/admin/Bert/chinese_L-12_H-768_A-12",
                "init_checkpoint": "bert_model.ckpt",
    
                "learning_rate": 1e-5,
                "max_seq_len": 256,
                "weight_decay": 0.0,
                "epochs": 10,
                "batch_size": 16,
                "dropout": 0.8,
                "folds": 0.8,
                }
    
    def __init__(self,
                 component_config = None,
                 sess = None,
                 graph = None,
                 vocabs = None,
                 tags = None):
        
        super(BiLSTMCrfEntityExtractor, self).__init__(component_config)
        
        self.normalizer = Normalizers().parse_dict_config(
            self.component_config.get("normalizers", {})
        )
        
        self.component_config["folds"] = min(self.component_config.get("folds", 1.), 1.)
        
        self.sess = sess
        self.g = graph
        self.vocabs = vocabs
        self.tags = tags
        
        self.eval_res = {}
    
    @staticmethod
    def get_config_proto():
        """training session config,
        GPU would auto determined"""
        
        gpus = getGPUs()
        gpus = getAvailabilityGPU(gpus)
        
        gpus = str(gpus.id) if gpus != "cpu" else ""
        
        config = tf.ConfigProto(
            device_count={"CPU": cpu_count(),
                          "GPU": 1 if gpus else 0},
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options={
                "allow_growth": True,
                "per_process_gpu_memory_fraction": 0.5,
                "visible_device_list": gpus
            }
        )
        
        return config
    
    def train(self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs) -> None:
        self.component_config = config.for_component(self.name, self.defaults)
        
        _tag_schema = self.component_config.get("tag_schema", "BIO").upper()
        _embedding = self.component_config.get("embedding", "embedding")
        
        train_examples, dev_examples = training_data.train_test_split(train_frac = self.component_config.get("folds", 1.))
        
        assert _tag_schema in ["BIO", "BIOES"], \
            "Only supported for `BIO` or `BIOES` tag schema"
        
        if train_examples.entity_examples:
            filtered_entity_examples = self.filter_trainable_entities(train_examples.training_examples)
            filtered_dev_entity_examples = self.filter_trainable_entities(dev_examples.training_examples)
            
            # get features, where the embedding method differs from bert
            if _embedding == "embedding":
                self.processor = self._load_embedding_processor(filtered_entity_examples,
                                                                filtered_dev_entity_examples,
                                                                **self.component_config)
                self._train_embedding()
            
            elif _embedding == "bert":
                self.processor = self._load_bert_processor(filtered_entity_examples,
                                                           filtered_dev_entity_examples,
                                                           **self.component_config)
                self._train_bert()
            
            else:
                raise ValueError("Unknown processor for given `embedding`,"
                                 "only `embedding` or `bert` received")
    
    
    def _load_embedding_processor(self, examples, dev_examples, tag_schema, filter_threshold, use_seg,
                                  initial_tokens, initial_tags, lower_case, **kwargs):
        """load embedding processor"""
        processor = VocabFeatureProcessor(tag_schema = tag_schema,
                                          filter_threshold = filter_threshold,
                                          initial_tokens = initial_tokens,
                                          initial_tags = initial_tags,
                                          lower_case = lower_case)
        train_index = len(examples)
        processor.create_examples(examples = examples + dev_examples,
                                  normalizer = self.normalizer)
        processor.create_features(use_seg = use_seg, examples = processor.examples[:train_index])
        processor.create_features(use_seg = use_seg, examples = processor.examples[train_index:],
                                  is_training = False)
        
        self.tokens_vocab = processor.Vocab
        self.tags_vocab = processor.Tags
        
        return processor
    
    def _load_bert_processor(self, examples, dev_examples, tag_schema, max_seq_len,
                             bert_path, initial_tags, lower_case, **kwargs):
        """load bert processor"""
        processor = BertFeatureProcessor(tag_schema=tag_schema,
                                         max_seq_len=max_seq_len,
                                         vocab_file=opt.join(bert_path, "vocab.txt"),
                                         bert_config=opt.join(bert_path, "bert_config.json"),
                                         initial_tags=initial_tags,
                                         lower_case=lower_case)
        processor.Tags.START = "[CLS]"
        processor.Tags.STOP = "[SEP]"
        
        processor.create_examples(examples=examples + dev_examples,
                                  normalizer=self.normalizer)
        train_index = len(examples)
        processor.create_features(examples = processor.examples[:train_index])
        processor.create_features(examples = processor.examples[train_index:],
                                  is_training = False)
        
        
        self.tags_vocab = processor.Tags
        
        return processor
    
    def _train_embedding(self):
        """training model for embedding method"""
        
        model = BiLSTM_CRF(num_token = self.processor.Vocab.size(),
                           num_tags = self.processor.Tags.size(),
                           **self.component_config)
        
        _bt_time = time()
        self.g = model.build_graph()
        logger.info(f"build graph done, cost time {time() - _bt_time}")
        
        with self.g.as_default():
            self.sess = tf.Session(config = self.get_config_proto(), graph = self.g)
            self.sess.run(tf.global_variables_initializer())
            
            min_step_loss = float("inf")
            min_step = (None, None)
            
            pbar = tqdm(range(self.component_config.get("epochs", 10)))
            
            for epoch in pbar:
                # training batcher
                for bid, batch_data in enumerate(self.processor.batcher(features = self.processor.train_features,
                                                                        batch_size = self.component_config.get("batch_size", 16))):
                    feed_dict = {model.input_ids: batch_data[0],
                                 model.seg_ids: batch_data[1],
                                 model.label_ids: batch_data[2],
                                 model.dropout: self.component_config.get("dropout", 0.8),
                                 model.if_training: True}

                    _, b_loss = self.sess.run([model.train_op, model.loss], feed_dict = feed_dict)
                    
                    pbar.set_postfix({"epoch": epoch,
                                      "batch_loss": f"{b_loss:.3f}"})
                    
                    if b_loss < min_step_loss:
                        min_step = (epoch, bid)
                        min_step_loss = b_loss
                
                # dev batcher and evaluation
                eval_op = nerEvalCounter()
                eval_true_tags = []
                eval_pred_tags = []

                for _, batch_data in enumerate(self.processor.batcher(features = self.processor.dev_features,
                                                                      batch_size = 1)):
                    _, pred_ids = self.sess.run([model.logits, model.pred_ids], feed_dict = {
                        model.input_ids: batch_data[0],
                        model.seg_ids: batch_data[1],
                        model.label_ids: batch_data[2],
                        model.dropout: 1.,
                        model.if_training: False
                    })
    
                    eval_true_tags.append(self.tags_vocab.get_taguence(batch_data[2]))
                    eval_pred_tags.append(self.tags_vocab.get_taguence(pred_ids))

                # ner evaluation
                eval_res = eval_op.run(eval_true_tags, eval_pred_tags)
                self.eval_res[epoch] = eval_res

            logger.info(f"Finished training BiLSTM-CRF policy, "
                        f"Minimum training batch loss at step {min_step[0]}-{min_step[1]},")
    
    
    def _train_bert(self):
        """training model for bert method"""
        
        model = BertBiLSTM_CRF(num_tags = self.processor.Tags.size(),
                               bert_config=opt.join(self.component_config.get("bert_path"),
                                                    "bert_config.json"),
                               **self.component_config)
        
        _bt_time = time()
        self.g = model.build_graph()
        logger.info(f"build graph done, cost time {time() - _bt_time}")
        
        with self.g.as_default():
            self.sess = tf.Session(config = self.get_config_proto(), graph = self.g)

            # load pretrained bert model weights
            tvars = tf.trainable_variables()
            init_checkpoint = opt.join(self.component_config.get("bert_path"),
                                       self.component_config.get("init_checkpoint"))

            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars,
                                                                                              init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            self.sess.run(tf.global_variables_initializer())

            min_step_loss = float("inf")
            min_step = (None, None)
            hooks = self.component_config.get("metrics", "f1")
            max_hook_step = (0, -float("inf"))
            min_hook_step = (0, float("inf"))

            pbar = tqdm(range(self.component_config.get("epochs", 10)))

            for epoch in pbar:
                for bid, batch_data in enumerate(self.processor.batcher(features = self.processor.train_features,
                                                                        batch_size = self.component_config.get("batch_size", 16))):
                    feed_dict = {model.input_ids: batch_data[0],
                                 model.input_masks: batch_data[1],
                                 model.input_type_ids: batch_data[2],
                                 model.label_ids: batch_data[3],
                                 model.dropout: self.component_config.get("dropout", 0.8),
                                 model.if_training: True}
                    _, b_loss = self.sess.run([model.train_op, model.loss], feed_dict = feed_dict)

                    pbar.set_postfix({"epoch": epoch,
                                      "batch_loss": f"{b_loss:.3f}"})

                    if b_loss < min_step_loss:
                        min_step = (epoch, bid)
                        min_step_loss = b_loss

                # dev batcher and evaluation
                eval_op = nerEvalCounter()
                eval_true_tags = []
                eval_pred_tags = []
                for batch_data in self.processor.batcher(features = self.processor.dev_features,
                                                         batch_size = 1):
                    pred_ids = self.sess.run(model.pred_ids, feed_dict = {model.input_ids: batch_data[0],
                                                                          model.input_masks: batch_data[1],
                                                                          model.input_type_ids: batch_data[2],
                                                                          model.label_ids: batch_data[3],
                                                                          model.dropout: 1.,
                                                                          model.if_training: False})
                    _seq_len = np.sum(batch_data[1])
                    eval_true_tags.append(self.tags_vocab.get_taguence(batch_data[3][:_seq_len]))
                    eval_pred_tags.append(self.tags_vocab.get_taguence(pred_ids[:_seq_len]))

                    # ner evaluation
                eval_res = eval_op.run(eval_true_tags, eval_pred_tags)
                self.eval_res[epoch] = eval_res
                if hooks not in eval_res:
                    logger.warning(f"Bert-BiLSTM-CRF policy assigned hooks {hooks} not in evaluation metrics"
                                   f"or Dev procedure not conducted")
                    continue

                if eval_res[hooks] > max_hook_step[1]:
                    max_hook_step = (epoch, eval_res[hooks])
                if eval_res[hooks] < min_hook_step[1]:
                    min_hook_step = (epoch, eval_res[hooks])

            logger.info(f"Finished training Bert-BiLSTM-CRF policy, "
                        f"Minimum training batch loss at step {min_step[0]}-{min_step[1]},"
                        f"Max Hook {hooks} Step {max_hook_step},"
                        f"Min Hook {hooks} Step {min_hook_step}")
    
    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> BiLSTMCrfEntityExtractor
        
        meta = model_metadata.for_component(cls.name)
        
        _embedding = meta.get("embedding")
        relative_dictionary_path = meta.get("bilstm_jection", None)
        
        if model_dir and relative_dictionary_path:
            ckpt = opt.join(model_dir, relative_dictionary_path, "bilstm_jection.ckpt")
            vocabs_f = opt.join(model_dir, relative_dictionary_path, "token2idx.pkl")
            tags_f = opt.join(model_dir, relative_dictionary_path, "idx2tag.pkl")
            
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(config = cls.get_config_proto())
                saver = tf.train.import_meta_graph(ckpt + '.meta')
                
                saver.restore(sess, ckpt)
            
            with open(vocabs_f, "rb") as f:
                vocabs = pickle.load(f)
            
            with open(tags_f, "rb") as f:
                tags = pickle.load(f)
            
            return cls(meta, sess = sess, graph = graph,
                       vocabs = vocabs, tags = tags)
        
        else:
            return cls(meta)
    
    def process(self, message: Message, **kwargs: Any):
        """pass"""
        
        if self.sess is None:
            logger.error("`BiLSTM-CRF projection model` not trained correctly,"
                         "components will pass and pipeline procedure continue")
            return
        
        res = []
        
        _embedding = self.component_config['embedding']
        
        r_text = message.text
        token = [x.text for x in message.get("tokens", [])]
        
        normalizer = self.normalizer
        lower = self.component_config["lower_case"]
        tag_schema = self.component_config["tag_schema"]
        
        text, words, tags = Processor()._convert_exam(text=r_text,
                                                      words=token,
                                                      ent_offsets=[],
                                                      tag_schema=tag_schema,
                                                      normalizer=normalizer,
                                                      lower=lower)

        if _embedding == "embedding":
            res, confidence = self._decode_embedding_entities(text, words, tags)
        elif _embedding == "bert":
            res, confidence = self._decode_bert_entities(text, words, tags)
        else:
            return
        
        # TODO: add confidence to result
        res = tagNormalizer(res).run()
        
        extracted = self.add_extractor_name(pred_result_to_json(r_text, res, confidence))
        
        message.set("entities", message.get("entities", []) + extracted, add_to_output = True)
    
    def _decode_embedding_entities(self, text, words, tags):
        """prediction for embedding method"""
        
        input_ids_place = self.g.get_tensor_by_name("input_ids:0")
        label_ids_place = self.g.get_tensor_by_name("label_ids:0")
        seg_ids_place = self.g.get_tensor_by_name("seg_ids:0")
        dropout_place = self.g.get_tensor_by_name("dropout:0")
        if_training_place = self.g.get_tensor_by_name("if_training:0")
        
        pred_ids_place = self.g.get_tensor_by_name("pred_ids:0")
        logits_place = self.g.get_tensor_by_name("logits:0")

        ds = namedtuple("dataset", ["text", "words", "tags"])
        ds_exam = ds(text, words, tags)
        fea = prepare_dataset(examples = [ds_exam],
                              vocabs = self.vocabs,
                              tags = self.tags,
                              use_seg = self.component_config["use_seg"])[0]

        pred_ids, logits = self.sess.run([pred_ids_place, logits_place],
                                         feed_dict = {input_ids_place: [fea.token_ids],
                                                      label_ids_place: [fea.tag_ids],
                                                      seg_ids_place: [fea.seg_ids],
                                                      dropout_place: 1.0,
                                                      if_training_place: False})
        
        pred_ids = np.squeeze(pred_ids).tolist()
        logits = softmax(np.squeeze(logits))

        confidence = logits[np.arange(len(text)), pred_ids].tolist()  # [seq_len]
        res = self.tags.get_taguence(pred_ids)  # [seq_len]
        
        return res, confidence
    
    def _decode_bert_entities(self, text, words, tags):
        """prediction for bert method"""
        input_ids_place = self.g.get_tensor_by_name("input_ids:0")
        input_masks_place = self.g.get_tensor_by_name("input_masks:0")
        input_type_ids_place = self.g.get_tensor_by_name("input_type_ids:0")
        label_ids_place = self.g.get_tensor_by_name("label_ids:0")
        dropout_place = self.g.get_tensor_by_name("dropout:0")
        if_training_place = self.g.get_tensor_by_name("if_training:0")
        pred_ids_place = self.g.get_tensor_by_name("pred_ids:0")
        logits_place = self.g.get_tensor_by_name("logits:0")
        
        
        exam = [InputExample(guid = "default",
                             text = text,
                             label = tags)]
        fea = prepare_bert_dataset(exam,
                                   tokenizer = self.vocabs,
                                   max_seq_len = self.component_config["max_seq_len"],
                                   tags = self.tags)[0]
        
        pred_ids, logits = self.sess.run([pred_ids_place, logits_place],
                                         feed_dict = {input_ids_place: [fea.input_ids],
                                                      input_masks_place: [fea.input_mask],
                                                      input_type_ids_place: [fea.input_type_ids],
                                                      label_ids_place: [fea.label_ids],
                                                      dropout_place: 1.0,
                                                      if_training_place: False})
        pred_ids = np.squeeze(pred_ids).tolist()[1:(len(text) + 1)] # [tag_size]
        logits = softmax(np.squeeze(logits)) # [seq_len, tag_size]
        
        confidence = logits[np.arange(1, len(text) + 1), pred_ids].tolist() # [seq_len]
        res = self.tags.get_taguence(pred_ids) # [seq_len]
        
        return res, confidence
    
    @staticmethod
    def _make_non_exist_dir(dirname):
        os.makedirs(dirname, exist_ok = True)
    
    @staticmethod
    def _copy_file(input_file, target_file):
        copyfile(input_file, target_file)
    
    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """pass"""
        
        if self.sess is None:
            return {"bilstm_jection": None}
        
        _target_model_dir = opt.join(model_dir, BILSTM_CRF_MODEL_NAME)
        self._make_non_exist_dir(_target_model_dir)
        _embedding = self.component_config.get("embedding", "embedding")
        
        write_to_file(opt.join(_target_model_dir, f"eval_{self.name}.txt"),
                      report_epoch_score_matr(self.eval_res))
        
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
            
            # save model separately
            saver.save(self.sess, opt.join(_target_model_dir, "bilstm_jection.ckpt"))
        
        with open(opt.join(_target_model_dir, "idx2tag.pkl"), "wb") as f:
            pickle.dump(self.processor.Tags, f)
        
        if _embedding == "embedding":
            # for embedding model
            # save model, vocab, tags,
            with open(opt.join(_target_model_dir, "token2idx.pkl"), "wb") as f:
                pickle.dump(self.processor.Vocab, f)
            
            return {"bilstm_jection": BILSTM_CRF_MODEL_NAME,
                    "vocab_size": self.processor.Vocab.size(),
                    "tag_size": self.processor.Tags.size()}
        
        elif _embedding == "bert":
            # for bert model
            # save model, tags, copy vocab/config file
            
            with open(opt.join(_target_model_dir, "token2idx.pkl"), "wb") as f:
                pickle.dump(self.processor.tokenizer, f)
            
            return {"bilstm_jection": BILSTM_CRF_MODEL_NAME,
                    "vocab_size": len(self.processor.tokenizer.vocab),
                    "tag_size": self.processor.Tags.size()}


