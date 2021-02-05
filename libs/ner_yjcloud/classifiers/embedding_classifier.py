"""
Usage: StarSpace Embedding Intent Classifier

An embedding intent classifier based on tensorflow
Based on `Starspace` from paper:
    `StarSpace: Embed All The Things!`
    https://arxiv.org/pdf/1709.03856.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import os.path as opt
from tqdm import tqdm
from multiprocessing import cpu_count

from typing import List, Text, Any, Optional, Dict, Tuple, Union, Set

from ner_yjcloud.models.start_space.utils import create_intent_dict, create_encoded_intent_bag
from ner_yjcloud.models.start_space.starspace import StarSpace
from ner_yjcloud.models import clsEvalCounter, report_epoch_score_matr

from ner_yjcloud.utils import write_to_file
from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data import TrainingData
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message

from ner_yjcloud.classifiers import INTENT_RANKING_LENGTH
from ner_yjcloud.components import Component
from ner_yjcloud.models.gpu_utils import getGPUs, getAvailabilityGPU
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import tensorflow as tf

except:
    raise ImportError("No tensorflow package found")

logger = logging.getLogger(__name__)


STARSPACE_NAME = "start_space_model"


class StarSpaceEmbeddingClassifier(Component):
    """pass"""

    name = "start_embedding_intent_classifier"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "num_hidden_layers_a": 2,
        "hidden_layer_size_a": [256, 128],
        "num_hidden_layers_b": 0,
        "hidden_layer_size_b": [],
        "batch_size": 32,
        "epochs": 100,
        "folds": 0.8,

        # embedding parameters
        "embed_dim": 20,
        "mu_pos": 0.8,
        "mu_neg": -0.4,
        "similarity_type": "cosine",
        "num_neg": 20,
        "use_max_sim_neg": True,

        # regularization
        "C2": .002,
        "C_emb": .8,
        "dropout_rate": .8,

        # flag if tokenize intents
        "intent_split_symbol": "_",

        # visualization of accuracy
        "evaluate_every_num_epochs": 10,
    }

    @classmethod
    def required_packages(cls):  # type: () -> List[Text]
        return ["tensorflow"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 session = None,
                 graph = None,
                 idx2int: Dict = None,
                 encoded_intents_bag = None,
                 text_input_dim = None,
                 intent_input_dim = None,
                 ):
        super(StarSpaceEmbeddingClassifier, self).__init__(component_config)

        self.session = session
        self.graph = graph
        self.idx2int = idx2int
        self.encoded_intents_bag = encoded_intents_bag

        self.folds = min(max(self.component_config["folds"], 0.2), 1.)

        self.text_input_dim = text_input_dim
        self.intent_input_dim = intent_input_dim
        
        self.eval_res = {}


    @staticmethod
    def get_config_proto():
        """training session config,
        GPU would auto determined"""

        gpus = getGPUs()
        gpus = getAvailabilityGPU(gpus)

        gpus = str(gpus.id) if gpus != "cpu" else ""

        config = tf.ConfigProto(
            device_count={"GPU": 1 if gpus else 0,
                          "CPU": cpu_count(),
                          },

            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options={
                "allow_growth": True,
                "per_process_gpu_memory_fraction": 0.5,
                "visible_device_list": gpus
            }
        )

        return config

    def create_vocabs(self, training_data: TrainingData) -> None:
        """
        create vocab for
            * intent to index vocab
            * index to intent vocab
            * intent bag vector
        """

        symbols = self.component_config.get("intent_split_symbol", "")

        intents = set([x.get("intent", "default") for x in training_data.intent_examples])

        self.int2idx = create_intent_dict(intents)

        if len(self.int2idx) < 2:
            raise ValueError("Intent sparse error, Need at least 2 different classes")

        self.idx2int = {val: key for key, val in self.int2idx.items()}
        self.encoded_intents_bag = create_encoded_intent_bag(self.int2idx, symbols)

    def _prepare_feature(self, training_data: TrainingData):
        """obtain single dataset features"""

        x_features = []
        intent_ids = []
        y_labels = []
        total_size = 0

        for exam in training_data.intent_examples:
            x_features.append(exam.get("text_features"))
            intent_id = self.int2idx[exam.get("intent", "default")]
            intent_ids.append(intent_id)

            y_labels.append(self.encoded_intents_bag[intent_id])

            total_size += 1


        if total_size:
            features = {
                "x": np.stack(x_features).squeeze(),
                "y": np.stack(y_labels),
                "intent": np.stack(intent_ids)
            }
        else:
            features = {
                "x": np.array([]),
                "y": np.array([]),
                "intent": np.array([])
            }

        return features, total_size

    def prepare_dataset(self, training_data: TrainingData):
        """obtain training dataset and dev dataset features feed into model"""


        train_examples, test_examples = training_data.train_test_split(train_frac = self.folds)

        train_features, train_size = self._prepare_feature(train_examples)
        test_features, test_size = self._prepare_feature(test_examples)

        return train_features, test_features, train_size

    def eval_feature(self, feature, size):
        """obtain evaluation examples features,
        where negative sample would not used"""

        choice_index = np.random.permutation(size)

        x = feature["x"][choice_index]
        # y = feature["y"][choice_index]
        intent = feature["intent"][choice_index]

        candi_y = np.stack([self.encoded_intents_bag for _ in range(x.shape[0])])

        return {"x": x,
                "y": candi_y,
                "intent": intent}

    def negative_feature(self, feature):
        """obtain training examples equipped with negative features"""
        x = feature["x"]
        y = feature["y"]
        intent = feature["intent"]

        y = y[:, np.newaxis, :] # [batch, 1, len_int]

        neg_y = np.zeros((y.shape[0], self.component_config["num_neg"], y.shape[2]))
        for idx in range(y.shape[0]):
            neg_idx = [i for i in range(self.encoded_intents_bag.shape[0]) if i != intent[idx]]
            negs = np.random.choice(neg_idx, size = self.component_config["num_neg"])

            neg_y[idx] = self.encoded_intents_bag[negs]

        new_y = np.concatenate([y, neg_y], 1) # [batch, neg + 1, len_int]

        return {"x": x,
                "y": new_y,
                "intent": intent}

    def batcher(self, features, size, batch_size):
        """pass"""

        index = np.random.choice(size, size = size, replace = False)
        features = {"x": features["x"][index],
                    "y": features["y"][index],
                    "intent": features["intent"][index]}

        for id in range(0, size, batch_size):
            yield {"x": features["x"][id: min(id + batch_size, size)],
                   "y": features["y"][id: min(id + batch_size, size)],
                   "intent": features["intent"][id: min(id + batch_size, size)]}


    def train(self,
              training_data: TrainingData,
              config: Optional[RasaNLUModelConfig],
              **kwargs: Any) -> None:
        """pass"""
        self.component_config = config.for_component(self.name, self.defaults)

        self.create_vocabs(training_data)

        training_features, test_features, training_size = self.prepare_dataset(training_data)

        self.text_input_dim = training_features["x"].shape[-1]
        self.intent_input_dim = training_features["y"].shape[-1]


        model = StarSpace(text_dim = self.text_input_dim,
                          intent_dim = self.intent_input_dim,
                          **self.component_config)

        self.graph = model.build_graph()
        batch_size = self.component_config["batch_size"]
        epochs = self.component_config["epochs"]
        ep_visual = self.component_config["evaluate_every_num_epochs"]
        dropout = min(max(self.component_config["dropout_rate"], 0.5), 1.)

        pbar = tqdm(total = epochs, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        b_loss = 0.

        with self.graph.as_default():
            self.session = tf.Session(config = self.get_config_proto(),
                                      graph = self.graph)
            self.session.run(tf.global_variables_initializer())

            for ep in range(epochs):
                for batch_data in self.batcher(training_features, training_size, batch_size):
                    # get negative features
                    batch_data = self.negative_feature(batch_data)

                    _, b_loss, b_pred = self.session.run([model.train_op, model.loss, model.pred_ids],
                                                         feed_dict = {model.text_in: batch_data["x"],
                                                                      model.intent_in: batch_data["y"],
                                                                      model.label_in: batch_data["intent"],
                                                                      model.dropout: dropout})
                    pbar.set_postfix({"ep": ep,
                                      "loss": f"{b_loss:.3f}"})

                # evaluate
                if ep_visual > 0 and ep % ep_visual == 0:
                    eval_res = self.evaluate(model,
                                             test_features)
                    self.eval_res[ep] = eval_res

                pbar.update(1)

            logger.info(f"Finished training StarSpace Embedding Intent policy,"
                        f"with last training batch loss {b_loss:.3f}")

    def evaluate(self, model, test_features):
        """pass"""
        eval_res = {}

        if test_features["x"].size != 0:
            test_eval_feature = self.eval_feature(test_features, test_features["x"].shape[0])
    
            eval_pred_ids = self.session.run(model.pred_ids,
                                             feed_dict={model.text_in: test_eval_feature["x"],
                                                        model.intent_in: test_eval_feature["y"],
                                                        model.label_in: test_eval_feature["intent"],
                                                        model.dropout: 1.})
            eval_res = clsEvalCounter(self.idx2int).run(test_eval_feature["intent"],
                                                        eval_pred_ids)

        return eval_res


    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> BiLSTMCrfEntityExtractor

        meta = model_metadata.for_component(cls.name)

        relative_dictionary_path = meta.get("model_name", None)

        if model_dir and relative_dictionary_path:
            ckpt = opt.join(model_dir, relative_dictionary_path, "star_space.ckpt")
            init_text_dim = meta["init_text_dim"]
            init_intent_dim = meta["init_intent_dim"]


            with open(opt.join(model_dir, relative_dictionary_path, "intent_vocab.pkl"), "rb") as f:
                intent_vocab = pickle.load(f)
                idx2int = intent_vocab["idx2int"]
                encoded_intent = intent_vocab["encoded_intent"]

            model = StarSpace(text_dim = init_text_dim,
                              intent_dim = init_intent_dim, **meta)

            graph = tf.Graph()

            with graph.as_default():
                sess = tf.Session(config = cls.get_config_proto(), graph = graph)
                saver = tf.train.import_meta_graph(ckpt + '.meta')
                saver.restore(sess, ckpt)

                return cls(meta,
                           session = sess,
                           graph = graph,
                           idx2int = idx2int,
                           encoded_intents_bag = encoded_intent,
                           text_input_dim = init_text_dim,
                           intent_input_dim = init_intent_dim)

        else:
            return cls(meta)


    def process(self, message: Message, **kwargs: Any):
        """pass"""
        intent = {"name": None, "confidence": 0.}
        intent_ranking = []

        if self.graph is None:
            logger.error("No model loaded")

        else:
            feature = {"x": message.get("text_features").reshape(1, -1),
                       "y": np.stack([self.encoded_intents_bag for _ in range(1)]),
                       "intent": np.array([0])}

            sim, a, b = self.session.run([self.graph.get_tensor_by_name("sim:0"),
                                          self.graph.get_tensor_by_name("text_embed:0"),
                                          self.graph.get_tensor_by_name("intent_embed:0")],
                                         feed_dict = {self.graph.get_tensor_by_name("text_in:0"): feature["x"],
                                                      self.graph.get_tensor_by_name("intent_in:0"): feature["y"],
                                                      self.graph.get_tensor_by_name("label_in:0"): feature["intent"],
                                                      self.graph.get_tensor_by_name("dropout:0"): 1.})

            if self.component_config["similarity_type"] == "cosine":
                sim[sim < 0.] = 0.

            elif self.component_config["similarity_type"] == "inner":
                sim = np.exp(sim) / np.sum(sim)

            sim = sim.flatten()
            intent_ids = np.argsort(sim)[::-1]

            intent = {"name": self.idx2int[intent_ids[0]],
                      "confidence": sim[intent_ids[0]].tolist()}

            intent_ranking = [{"name": self.idx2int[intent_ids[idx]],
                               "confidence": sim[idx].tolist()} for idx in intent_ids[:INTENT_RANKING_LENGTH]]

        message.set("intent", intent, add_to_output = True)
        message.set("intent_ranking", intent_ranking, add_to_output = True)


    @staticmethod
    def create_folder(model_dir):
        """pass"""
        os.makedirs(model_dir, exist_ok = True)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        if self.session is None:
            return {"model_name": None,
                    "init_text_dim": None,
                    "init_intent_dim": None}

        target_path = opt.join(model_dir, STARSPACE_NAME)

        self.create_folder(target_path)

        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
            saver.save(self.session, opt.join(target_path, "star_space.ckpt"))

        with open(opt.join(target_path, "intent_vocab.pkl"), "wb") as f:
            pickle.dump({
                "idx2int": self.idx2int,
                "encoded_intent": self.encoded_intents_bag
            }, f)

        write_to_file(opt.join(target_path, f"eval_{self.name}.txt"),
                      report_epoch_score_matr(self.eval_res))

        return {"model_name": STARSPACE_NAME,
                "init_text_dim": self.text_input_dim,
                "init_intent_dim": self.intent_input_dim}

