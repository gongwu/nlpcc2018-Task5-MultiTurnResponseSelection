# -*- coding:utf-8 _*-
from __future__ import print_function
import tensorflow as tf
from models.SCN_model import SCNModel
from utils import tf_utils


class SCNMAModel(SCNModel):
    def __init__(self, config, data):
        super(SCNMAModel, self).__init__(config, data)

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
        # A_matrix bilinear
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.hidden_size_1, self.hidden_size_1),
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.hidden_size_2, kernel_initializer=tf.orthogonal_initializer())
        attention_GRU = tf.nn.rnn_cell.GRUCell(self.hidden_size_2, kernel_initializer=tf.orthogonal_initializer())
        reuse = None
        # b * s * o
        response_GRU_embeddings, response_last_hidden = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings,
                                                       sequence_length=self.input_res_len, dtype=tf.float32,
                                                       scope='response_sentence_GRU')
        self.response_embedding_save = response_GRU_embeddings
        # b * d * s
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        # b * o * s
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        utter_respresent = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            # [b * s * d] [b * d * s] = [b * s * s]
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, utterance_last_hidden = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings,
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
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse,
                                              name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
            utter_respresent.append(utterance_last_hidden)
        # b * u * 50
        memory_vectors = tf.stack(matching_vectors, axis=1, name='matching_stack')
        utter_respresent = tf.stack(utter_respresent, axis=1, name="utter_respresent")
        match_size = memory_vectors.get_shape()[2].value
        attention_vectors = []
        steps = 0
        reuse = None
        while steps < 5:
            if steps > 0:
                reuse = True
            # b * 2s
            attention = self.self_attention(memory_vectors, match_size, self.hidden_size_1, response_last_hidden,
                                            self.input_utter_num, self.utter_len, scope="memory", reuse=reuse)
            # attention = self.self_attention(utter_respresent, self.hidden_size_1, self.hidden_size_1,
            #                                 response_last_hidden,
            #                                 self.input_utter_num, self.utter_len, scope="memory", reuse=reuse)
            attention_vectors.append(attention)
            steps += 1
        # b * u * 50
        attention_vectors = tf.stack(attention_vectors, axis=1, name='attention_stack')
        _, attention_last_hidden = tf.nn.dynamic_rnn(attention_GRU, attention_vectors,
                                                     dtype=tf.float32, scope='attention_GRU')
        _, final_last_hidden = tf.nn.dynamic_rnn(final_GRU, memory_vectors, sequence_length=self.input_utter_num,
                                                 dtype=tf.float32, scope='final_GRU')  # TODO: check time_major
        if self.config.mode == "SCNMA":
            last_hidden = tf.concat([attention_last_hidden, final_last_hidden], axis=-1, name='last_hidden_stack')
        elif self.config.mode == "SCN":
            last_hidden = final_last_hidden
        elif self.config.mode == "MEMORY":
            last_hidden = attention_last_hidden
        logits = tf.layers.dense(last_hidden, self.num_class, kernel_initializer=tf.contrib.layers.xavier_initializer(),
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

    def self_attention(self, match_state, match_size, hidden_size, context_vecotor, utter_num, max_utter_num, scope=None, reuse=None):
        """
        hidden_state: [batch_size, sequence_length, hidden_size*2]
        context vector:
        :return [batch_size*num_sentences, hidden_size*2]
        """
        with tf.variable_scope(scope or "self_attention", reuse=reuse):
            self.W_w_m = tf.get_variable("W_w_m", shape=[match_size, match_size])
            self.W_b_m = tf.get_variable("W_b_m", shape=[match_size])
            self.W_w_h = tf.get_variable("W_w_h", shape=[hidden_size, match_size])
            self.W_b_h = tf.get_variable("W_b_h", shape=[match_size])
            # self.context_vecotor_word = tf.get_variable("informative_word_attention", shape=[hidden_size])  # TODO o.k to use batch_size in first demension?
            # 0) one layer of feed forward network
            # shape: [batch_size*sequence_length, hidden_size]
            match_state_ = tf.reshape(match_state, shape=[-1, match_size])
            # hidden_state_: [batch_size*sequence_length, hidden_size]
            # W_w_attention_sentence: [hidden_size, hidden_size]
            match_representation = tf.nn.tanh(tf.matmul(match_state_, self.W_w_m)
                                               + self.W_b_m)
            context_vecotor = tf.nn.tanh(tf.matmul(context_vecotor, self.W_w_h) + self.W_b_h)
            # shape: [batch_size, sequence_length, hidden_size]
            match_representation = tf.reshape(match_representation, shape=[-1, max_utter_num, match_size])
            context_vecotor = tf.expand_dims(context_vecotor, axis=1)
            # 1) get logits for each word in the sentence.
            # hidden_representation: [batch_size, sequence_length, hidden_size]
            # context_vecotor_word: [hidden_size]
            hidden_state_context_similiarity = tf.multiply(match_representation, context_vecotor)
            # 对应相乘再求和，得到权重
            # shape: [batch_size, sequence_length]
            attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
            # subtract max for numerical stability (softmax is shift invariant).
            # tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
            # shape: [batch_size, 1]
            attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
            # 2) get possibility distribution for each word in the sentence.
            # shape: [batch_size, sequence_length]
            # 归一化
            p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
            # 3) get weighted hidden state by attention vector
            # shape: [batch_size, sequence_length, 1]
            p_attention_expanded = tf.expand_dims(p_attention, axis=2)
            # below sentence_representation
            # shape:[batch_size, sequence_length, hidden_size]<----
            # p_attention_expanded: [batch_size, sequence_length, 1]
            # hidden_state_: [batch_size, sequence_length, hidden_size]
            # shape: [batch_size, sequence_length, hidden_size]
            sentence_representation = tf.multiply(p_attention_expanded, match_representation)
            # shape: [batch_size,sequence_length, hidden_size]
            sentence_representation = tf_utils.Mask(sentence_representation, utter_num, max_utter_num)
            sentence_representation = tf.reduce_sum(sentence_representation, axis=1)
            # shape: [batch_size, hidden_size]
            return sentence_representation



