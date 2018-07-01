# -*- coding:utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from base.base_model import BaseModel
from utils import tf_utils


class SCNModel(BaseModel):
    def __init__(self, config, data):
        super(SCNModel, self).__init__(config, data)
        # Add PlaceHolder
        self.hidden_size_1 = 200
        self.hidden_size_2 = 50
        self.num_class = self.config.num_class
        self.utter_len = self.config.max_utter_len
        self.input_x_utter = tf.placeholder(tf.int32, shape=(None, self.utter_len, self.seq_len))
        self.input_x_response = tf.placeholder(tf.int32, shape=(None, self.seq_len))
        self.input_x_feature_p = tf.placeholder(tf.float32, shape=(None, 22))
        self.input_x_feature_u = tf.placeholder(tf.float32, shape=(None, self.utter_len, 22))
        self.input_res_len = tf.placeholder(tf.int32, shape=(None,))
        self.input_utter_len = tf.placeholder(tf.int32, shape=(None, self.utter_len))
        self.input_utter_num = tf.placeholder(tf.int32, shape=(None,))
        self.input_y = tf.placeholder(tf.int32, shape=(None, self.num_class))
        self.attention_num = self.config.attention_num
        self.weighted_mask = tf.placeholder(tf.float32, shape=(None, self.utter_len))
        self.enc_padding_mask = tf.placeholder(tf.float32, [None, self.config.max_utter_len], name='enc_padding_mask')
        self.final_padding_mask = tf.placeholder(tf.float32, [None, self.config.max_utter_len], name='final_padding_mask')
        self.rand_unif_init = tf.random_uniform_initializer(-self.config.rand_unif_init_mag,
                                                            self.config.rand_unif_init_mag, seed=123)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.config.trunc_norm_init_std)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # b * u * s * d
        all_utterance_embeddings = tf.nn.embedding_lookup(self.we, self.input_x_utter)
        # b * s * d
        response_embeddings = tf.nn.embedding_lookup(self.we, self.input_x_response)
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.hidden_size_1, kernel_initializer=tf.orthogonal_initializer())
        # u * b * s * d
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.utter_len, axis=1)
        # u * b
        all_utterance_len = tf.unstack(self.input_utter_len, num=self.utter_len, axis=1)
        if self.config.feature_u:
            # u * b * 22
            all_utterance_feature = tf.unstack(self.input_x_feature_u, num=self.utter_len, axis=1)
        # A_matrix bilinear
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.hidden_size_1, self.hidden_size_1),
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.hidden_size_2, kernel_initializer=tf.orthogonal_initializer())
        reuse = None
        # b * s * o
        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings,
                                                       sequence_length=self.input_res_len, dtype=tf.float32,
                                                       scope='response_sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        # b * d * s
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        # b * o * s
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        # for utterance_embeddings, utterance_len, utterance_feature in zip(all_utterance_embeddings, all_utterance_len, all_utterance_feature):
        for i in range(len(all_utterance_embeddings)):
            utterance_embeddings = all_utterance_embeddings[i]
            utterance_len = all_utterance_len[i]
            if self.config.feature_u:
                utterance_feature = all_utterance_feature[i]
            # [b * s * d] [b * d * s] = [b * s * s]
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings,
                                                            sequence_length=utterance_len,
                                                            dtype=tf.float32,
                                                            scope='utterance_sentence_GRU')
            # [b * s * o] * [o * o] = [b * s * o]
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            # [b * s * o] * [b * o * s] = [b * s * s]
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            # b * 50
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse,
                                              name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            if self.config.feature_u:
                matching_vector = tf.concat([matching_vector, utterance_feature], axis=-1)
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(all_utterance_feature, axis=0, name='matching_stack'),
                                           sequence_length=self.input_utter_num, dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major
        if self.config.feature_p:
            last_hidden = tf.concat([last_hidden, self.input_x_feature_p], axis=-1)
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_v')
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, 1), tf.int32)

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            loss = tf.reduce_mean(loss)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.get_shape().ndims > 1])
            reg_loss = loss + self.config.lambda_l2 * l2_loss
            # Build the loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            if self.config.clipper:
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.clipper)
                train_step = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=self.global_step_tensor)
            else:
                train_step = optimizer.minimize(loss, global_step=self.global_step_tensor)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.logits = logits
        self.loss = loss
        self.reg_loss = reg_loss
        self.train_step = train_step
