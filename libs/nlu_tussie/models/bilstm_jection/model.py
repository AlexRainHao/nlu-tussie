"""
tensorflow based model including
    * Embed_CRF
    * BiLSTM_CRF
    * Bert_CRF
    * Bert_BiLSTM_CRF
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.nn as nn

from ..bert.modeling import BertModel, BertConfig

__all__ = ["BiLSTM_CRF", "BertBiLSTM_CRF"]

class BiLSTM_CRF:
    """pass"""

    def __init__(self, token_dim, seg_dim,
                 num_token, num_tags,
                 hidden_dim, learning_rate,
                 weight_decay = 0.,
                 use_seg = True,
                 crf_only = False,
                 **kwargs):


        self.token_dim = token_dim
        self.seg_dim = seg_dim
        self.num_token = num_token
        self.num_tags = num_tags

        self.hidden_dim = hidden_dim
        self.learning_rate = float(learning_rate)
        self.weight_decay = weight_decay

        self.use_seg_feature = use_seg
        self.crf_only = crf_only

        self.embedding_dim = token_dim + (seg_dim if use_seg else 0)

    def build_graph(self):
        """pass"""

        g = tf.Graph()

        with g.as_default():

            self._create_placeholders()
            self._create_embedding_layers()
            self._create_feature_layer()
            self._create_crf_layer()
            self._create_decode_layer()
            self._create_train_layer()

        print("create projection BiLSTM Crf graph done")

        return g

    def _create_placeholders(self):
        """pass"""
        self.input_ids = tf.placeholder(tf.int32, shape = [None, None], name = "input_ids")
        self.label_ids = tf.placeholder(tf.int32, shape = [None, None], name = "label_ids")

        # if self.use_seg_feature:
        self.seg_ids = tf.placeholder(tf.int32, shape = [None, None], name = "seg_ids")

        self.if_training = tf.placeholder(tf.bool, shape = (), name = "if_training")
        self.dropout = tf.placeholder(tf.float32, name = "dropout")

        self.length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.input_ids)), reduction_indices = 1), tf.int32)

    def _create_embedding_layers(self):
        """pass"""
        embedding = []

        with tf.device("/cpu:0"), tf.variable_scope("embedding_layer"):
            self.token_embedding_layer = tf.get_variable("token_embedding",
                                                         shape = [self.num_token, self.token_dim],
                                                         initializer = tc.layers.xavier_initializer())
            embedding.append(nn.embedding_lookup(self.token_embedding_layer,
                                                 self.input_ids))

            if self.use_seg_feature:
                self.seg_embedding_layer = tf.get_variable("seg_embedding",
                                                           shape = [4, self.seg_dim],
                                                           initializer = tc.layers.xavier_initializer())
                embedding.append(nn.embedding_lookup(self.seg_embedding_layer,
                                                     self.seg_ids))

            embedding = tf.concat(embedding, axis = -1)

            assert embedding.shape[-1].value == self.embedding_dim, \
                f"concatenated embedding shape {embedding.shape} error"

            self.embedding = nn.dropout(embedding, self.dropout)

    def _create_feature_layer(self):
        """pass"""
        if self.crf_only:
            outs = self._projection_crf_layer()

        else:
            outs = self._projection_lstm_layer()

        self.logits = tf.identity(outs, name = "logits")

    def _create_crf_layer(self):
        """pass"""
        with tf.variable_scope("crf_loss_layer"):
            transitions = tf.get_variable("transitions",
                                          shape = [self.num_tags, self.num_tags],
                                          initializer = tc.layers.xavier_initializer())


            def _pass_fn():
                loss = tf.constant(0.0, name = "loss")
                return loss

            def _conti_fn(transitions=None):
                ll, transitions = tc.crf.crf_log_likelihood(
                    inputs = self.logits,
                    tag_indices = self.label_ids,
                    transition_params = transitions,
                    sequence_lengths = self.length
                )
                loss = tf.reduce_mean(-ll, name = "loss")

                if self.weight_decay > 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                    loss += self.weight_decay * l2_loss

                return loss

            loss = tf.cond(self.if_training, false_fn = lambda: _pass_fn(), true_fn = lambda: _conti_fn(transitions))
            self.loss = loss
            self.transitions = transitions

    def _create_decode_layer(self):
        pred_ids, _ = tc.crf.crf_decode(potentials = self.logits,
                                        transition_params = self.transitions,
                                        sequence_length = self.length)
        self.pred_ids = tf.identity(pred_ids, name = "pred_ids")

    def _create_train_layer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def _projection_crf_layer(self):
        """pass"""
        with tf.variable_scope("projection_crf_layer"):
            w = tf.get_variable("W", shape = [self.embedding_dim, self.num_tags],
                                dtype = tf.float32, initializer = tc.layers.xavier_initializer())

            b = tf.get_variable("b", shape = [self.num_tags], dtype = tf.float32,
                                initializer = tf.zeros_initializer())

            output = nn.xw_plus_b(self.embedding, w, b)

            return output

    def _projection_lstm_layer(self):
        """pass"""
        with tf.variable_scope("projection_lstm_layer"):
            cells = {
                "fw": nn.rnn_cell.DropoutWrapper(
                    nn.rnn_cell.LSTMCell(num_units = self.hidden_dim,
                                         initializer = tc.layers.xavier_initializer(),
                                         state_is_tuple = True),
                    output_keep_prob = self.dropout),
                "bw": nn.rnn_cell.DropoutWrapper(
                    nn.rnn_cell.LSTMCell(num_units = self.hidden_dim,
                                         initializer = tc.layers.xavier_initializer(),
                                         state_is_tuple = True),
                    output_keep_prob = self.dropout)}

            outputs, state = nn.bidirectional_dynamic_rnn(
                cell_fw = cells["fw"],
                cell_bw = cells["bw"],
                inputs = self.embedding,
                sequence_length = self.length,
                dtype = tf.float32
            )
            outputs = tf.concat(outputs, axis = 2)

            w = tf.get_variable("W",
                                shape = [self.hidden_dim * 2, self.num_tags],
                                dtype = tf.float32,
                                initializer = tc.layers.xavier_initializer())
            b = tf.get_variable("b", shape = [self.num_tags], dtype = tf.float32,
                                initializer = tf.zeros_initializer())

            outputs = tf.nn.xw_plus_b(outputs, w, b)

        return outputs


class BertBiLSTM_CRF(BiLSTM_CRF):
    """pass"""

    def __init__(self, num_tags,
                 hidden_dim,
                 learning_rate,
                 bert_config,
                 num_token = 0,
                 use_one_hot_embeddings = False,
                 token_dim = None, seg_dim = None,
                 weight_decay = 0.,
                 use_seg_feature = True, crf_only = False, **kwargs):
        super(BertBiLSTM_CRF, self).__init__(token_dim, seg_dim, num_token, num_tags,
                                             hidden_dim, learning_rate, weight_decay,
                                             use_seg_feature, crf_only)
        
        if isinstance(bert_config, str):
            self.bert_config = BertConfig.from_json_file(bert_config)
        elif isinstance(bert_config, BertConfig):
            self.bert_config = bert_config

        self.use_one_hot_embeddings = use_one_hot_embeddings

    def _create_placeholders(self):
        """pass"""
        self.input_ids = tf.placeholder(tf.int32, shape = [None, None], name = "input_ids")
        self.input_masks = tf.placeholder(tf.int32, shape = [None, None], name = "input_masks")
        self.input_type_ids = tf.placeholder(tf.int32, shape = [None, None], name = "input_type_ids")
        self.label_ids = tf.placeholder(tf.int32, shape = [None, None], name = "label_ids")

        self.if_training = tf.placeholder(tf.bool, shape = (), name = "if_training")

        # self.length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.input_ids)), reduction_indices = 1), tf.int32)
        self.length = tf.cast(tf.reduce_sum(self.input_masks, reduction_indices = 1), tf.int32)

        self.dropout = tf.placeholder(tf.float32, name = "dropout")


    def _create_embedding_layers(self):
        model = BertModel(config = self.bert_config,
                          is_training = self.if_training,
                          input_ids = self.input_ids,
                          input_mask = self.input_masks,
                          token_type_ids = self.input_type_ids,
                          use_one_hot_embeddings = self.use_one_hot_embeddings)
        self.embedding = model.get_sequence_output() # [batch, seq_len, embed_size]
        self.embedding_dim = self.embedding.shape[-1].value