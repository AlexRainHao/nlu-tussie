"""pass"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

class StarSpace:
    """pass"""

    def __init__(self,
                 text_dim, intent_dim,
                 num_hidden_layers_a, hidden_layer_size_a,
                 num_hidden_layers_b, hidden_layer_size_b,
                 embed_dim, mu_pos, mu_neg, num_neg,
                 C2, C_emb,
                 similarity_type, use_max_sim_neg,
                 **kwargs
                 ):

        for name, val in locals().items():
            if name == "kwargs" or name == "self":
                continue
            self.__setattr__(name, val)

        self.num_hidden_layers_a, self.hidden_layer_size_a = self._check_hidden_layer_size(self.num_hidden_layers_a,
                                                                                           self.hidden_layer_size_a,
                                                                                           prefix = "layer_a")
        self.num_hidden_layers_b, self.hidden_layer_size_b = self._check_hidden_layer_size(self.num_hidden_layers_b,
                                                                                           self.hidden_layer_size_b,
                                                                                           prefix = "layer_b")

    @staticmethod
    def _check_hidden_layer_size(num_layer, layer_size, prefix):
        if num_layer < 0:
            num_layer = 0

        if isinstance(layer_size, list) and len(layer_size) != num_layer:
            if len(layer_size) == 0:
                raise ValueError(f"{prefix} hidden layer size {layer_size} not mapping to layer num {num_layer}")

            layer_size = layer_size[0]

        if not isinstance(layer_size, list):
            layer_size = [layer_size for _ in range(num_layer)]

        return num_layer, layer_size


    def build_graph(self):
        """pass"""

        g = tf.Graph()

        with g.as_default():
            self._create_placeholders()
            self._create_embedding_layers()
            self._create_similarity_layer()
            self._create_loss_layer()
            self._create_train_layer()

        print("create star space model graph done")

        return g

    def _create_placeholders(self):
        """pass"""
        self.text_in = tf.placeholder(tf.float32, [None, self.text_dim], name = 'text_in')
        self.intent_in = tf.placeholder(tf.float32, [None, None, self.intent_dim], name = "intent_in")
        self.label_in = tf.placeholder(tf.float32, [None], name = "label_in")

        self.dropout = tf.placeholder(tf.float32, name = "dropout")

    def _create_embedding_single_layers(self, x_in, num_layer, num_size, prefix_name):
        """pass"""
        reg = tf.contrib.layers.l2_regularizer(self.C2)

        x = x_in

        for layer in range(num_layer):
            x = tf.layers.dense(inputs = x,
                                units = num_size[layer],
                                activation = tf.nn.relu,
                                reuse = tf.AUTO_REUSE,
                                kernel_regularizer = reg,
                                name = prefix_name + f'_{layer}'
                                )

            x = tf.nn.dropout(x, self.dropout)

        x = tf.layers.dense(inputs = x,
                            units = self.embed_dim,
                            reuse=tf.AUTO_REUSE,
                            kernel_regularizer = reg,
                            name = prefix_name + f"_embed"
                            )

        return x

    def _create_embedding_layers(self):
        """pass"""
        text_embed = self._create_embedding_single_layers(x_in = self.text_in,
                                                          num_layer = self.num_hidden_layers_a,
                                                          num_size = self.hidden_layer_size_a,
                                                          prefix_name = "embed_a_layer")

        intent_embed = self._create_embedding_single_layers(x_in = self.intent_in,
                                                            num_layer = self.num_hidden_layers_b,
                                                            num_size = self.hidden_layer_size_b,
                                                            prefix_name = "embed_b_layer")

        self.text_embed = tf.identity(text_embed, name = "text_embed")
        self.intent_embed = tf.identity(intent_embed, name = "intent_embed")

    def _create_similarity_layer(self):
        """pass"""

        if self.similarity_type == "cosine":
            self.norm_text_embed = tf.nn.l2_normalize(self.text_embed, -1)
            self.norm_intent_embed = tf.nn.l2_normalize(self.intent_embed, -1)

        if self.similarity_type in ["cosine", "inner"]:
            self.sim = tf.reduce_sum(tf.expand_dims(self.norm_text_embed, 1) * self.norm_intent_embed,
                                     axis = -1,
                                     name = "sim") # [batch, neg + 1]

            self.sim_embed = tf.reduce_sum(self.norm_intent_embed[:, 0:1, :] * self.norm_intent_embed[:, 1:, :],
                                           axis = -1,
                                           name = "sim_embed") # [batch, neg]

            self.pred_ids = tf.argmax(self.sim, -1, name = "pred_ids")

        else:
            raise ValueError("Wrong similarity type defined, "
                             "Only `cosine` or `inner` accepted")

    def _create_loss_layer(self):
        """pass"""
        if self.use_max_sim_neg:
            max_sim_neg = tf.reduce_max(self.sim[:, 1:], -1)
            loss = tf.reduce_mean(tf.maximum(0., self.mu_pos - self.sim[:, 0]) +
                                  tf.maximum(0., self.mu_neg + max_sim_neg))

        else:
            mu = self.mu_neg * np.ones(self.num_neg + 1)
            mu[0] = self.mu_pos

            factors = tf.concat([-1 * tf.ones([1, 1]),
                                 tf.ones([1, tf.shape(self.sim)[1] - 1])],
                                1) # [1, neg + 1]
            max_margin = tf.maximum(0., mu + factors * self.sim) # [batch, neg + 1]

            loss = tf.reduce_mean(tf.reduce_sum(max_margin, -1)) # [batch]

        max_sim_emb = tf.maximum(0., tf.reduce_max(self.sim_embed, -1))

        loss += tf.reduce_mean(max_sim_emb) * self.C_emb + tf.losses.get_regularization_loss()

        # if self.C2 > 0.:
        #     l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        #     loss += self.C2 * l2_loss

        self.loss = loss

    def _create_train_layer(self):
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)
