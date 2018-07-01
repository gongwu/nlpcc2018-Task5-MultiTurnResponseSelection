# -*- coding:utf-8 _*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
from models.SCNMA_model import SCNMAModel
from utils import tf_utils


class MBMNModel(SCNMAModel):
    def __init__(self, config, data):
        super(MBMNModel, self).__init__(config, data)

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
        # for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
        for i in range(len(all_utterance_embeddings)):
            utterance_embeddings = all_utterance_embeddings[i]
            utterance_len = all_utterance_len[i]
            if self.config.feature_u:
                utterance_feature = all_utterance_feature[i]
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
            if self.config.feature_u:
                matching_vector = tf.concat([matching_vector, utterance_feature], axis=-1)
            matching_vectors.append(matching_vector)
            utter_respresent.append(utterance_last_hidden)
        # b * u * 50
        matching_vectors = tf.stack(matching_vectors, axis=1, name='matching_stack')
        utter_respresent = tf.stack(utter_respresent, axis=1, name="utter_respresent")
        # b * u * h
        memory_vectors, final_last_hidden = tf.nn.dynamic_rnn(final_GRU, matching_vectors,
                                                              sequence_length=self.input_utter_num,
                                                              dtype=tf.float32,
                                                              scope='final_GRU')  # TODO: check time_major
        # v_memory_vectors = self.get_memory(memory_vectors, utter_respresent)
        query_vector = self.get_query(utter_respresent, self.final_padding_mask)
        self.s_weighted_mask = self.get_sim_weight(utter_respresent, query_vector, self.enc_padding_mask)
        self.combine_weight = self.get_combine_weight(self.weighted_mask, self.s_weighted_mask, self.enc_padding_mask)
        # l_memory_vectors = self.weighted_memory(v_memory_vectors, self.weighted_mask)
        # s_memory_vectors = self.weighted_memory(l_memory_vectors, self.s_weighted_mask)
        # c_memory_vectors = self.weighted_memory(v_memory_vectors, self.combine_weight)
        static_vector = self.static_weighted(memory_vectors)
        # with tf.variable_scope('self_attention'):
        #     attention = self.self_attention(v_memory_vectors, v_memory_vectors.get_shape()[2].value, scope="memory", reuse=None)
        dec_in_state = self.reduce_states(utterance_last_hidden)
        with tf.variable_scope('decoder'):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size_2, state_is_tuple=True, initializer=self.rand_unif_init)
            dec_out_state = self.add_decoder(dec_in_state, memory_vectors, response_last_hidden, cell)
        attention_last_hidden = dec_out_state.h
        if self.config.mode == "SCNRMA":
            last_hidden = tf.concat([attention_last_hidden, final_last_hidden], axis=-1, name='last_hidden_stack')
        elif self.config.mode == "SCN":
            last_hidden = static_vector
        elif self.config.mode == "MEMORY":
            last_hidden = attention_last_hidden
        if self.config.feature_p:
            last_hidden = tf.concat([last_hidden, self.input_x_feature_p], axis=-1)
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

    def reduce_states(self, last_hidden):
        encoder_hidden_dim = self.hidden_size_1
        decoder_hidden_dim = self.hidden_size_2
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [encoder_hidden_dim, decoder_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [encoder_hidden_dim, decoder_hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', decoder_hidden_dim, dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', decoder_hidden_dim, dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            # Apply linear layer
            # old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            # old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(last_hidden, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(last_hidden, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def get_memory(self, memory_vectors, utter_respresent):
        with tf.variable_scope('get_memory_vectors'):
            M_s = tf.get_variable('mem_gru_s', [1, 1, self.hidden_size_2, self.hidden_size_2], dtype=tf.float32,
                                  initializer=self.trunc_norm_init)
            W_s = tf.get_variable('hidden_gru_s', [1, 1, self.hidden_size_1, self.hidden_size_2], dtype=tf.float32,
                                  initializer=self.trunc_norm_init)
            b_s = tf.get_variable('bias_gate_s', self.hidden_size_2, dtype=tf.float32,
                                  initializer=self.trunc_norm_init)
            # (batch_size, s_length, 1, hidden_size)
            memory_features = tf.expand_dims(memory_vectors, axis=2)
            memory_features = tf.nn.conv2d(memory_features, M_s, [1, 1, 1, 1], "SAME")
            utter_features = tf.expand_dims(utter_respresent, axis=2)
            utter_features = tf.nn.conv2d(utter_features, W_s, [1, 1, 1, 1], "SAME")
            v_memory_vectors =tf.nn.tanh(tf.reduce_sum((memory_features + utter_features), 2) + b_s)
        return v_memory_vectors

    def get_memory_2(self, memory_vectors, utter_respresent):
        with tf.variable_scope('get_memory_vectors'):
            M_s = tf.get_variable('mem_gru_s', dtype=tf.float32,
                                  initializer=self.ortho_weight(self.hidden_size_2))
            M_s = tf.expand_dims(tf.expand_dims(M_s, 0), 0)
            W_s = tf.get_variable('hidden_gru_s', dtype=tf.float32,
                                  initializer=self.glorot_uniform((self.hidden_size_1, self.hidden_size_2)))
            W_s = tf.expand_dims(tf.expand_dims(W_s, 0), 0)
            b_s = tf.Variable(np.zeros(self.hidden_size_2, ), name='bias_gate_s', dtype=tf.float32)
            memory_features = tf.expand_dims(memory_vectors, axis=2)
            memory_features = tf.nn.conv2d(memory_features, M_s, [1, 1, 1, 1], "SAME")
            utter_features = tf.expand_dims(utter_respresent, axis=2)
            utter_features = tf.nn.conv2d(utter_features, W_s, [1, 1, 1, 1], "SAME")
            v_memory_vectors = tf.nn.tanh(tf.reduce_sum((memory_features + utter_features), 2) + b_s)
        return v_memory_vectors

    @staticmethod
    def get_query(memory_vectors, utter_final_mask):
        # b * 1 * s
        utter_final_mask = tf.expand_dims(utter_final_mask, axis=1)
        # [b, 1, s] * [b, s, h] = [b, 1 ,h]
        query_vector = tf.matmul(utter_final_mask, memory_vectors)
        # b * h
        query_vector = tf.reduce_sum(query_vector, 1)
        return query_vector

    @staticmethod
    def weighted_memory(v_memory_vectors, weighted_mask):
        weighted_mask = tf.expand_dims(weighted_mask, axis=2)
        v_memory_vectors *= weighted_mask
        return v_memory_vectors

    @staticmethod
    def get_combine_weight(l_weight, s_weight, utter_num_mask):
        combine_weight = l_weight * s_weight
        combine_weight *= utter_num_mask
        combine_weight /= tf.expand_dims(tf.reduce_sum(combine_weight, axis=-1), -1)
        return combine_weight

    def get_sim_weight(self, memory_vectors, query_vector, utter_num_mask):
        # b * h
        query_vector /= tf.expand_dims(tf.norm(query_vector, axis=-1), axis=-1)
        # b * 1 * h
        query_vector = tf.expand_dims(query_vector, axis=1)
        # b * s * h
        memory_vectors /= tf.expand_dims(tf.maximum(tf.norm(memory_vectors, axis=-1), self.config.eps), axis=-1)
        memory_vectors *= tf.expand_dims(utter_num_mask, axis=-1)
        # b * s
        # weighted_mask = tf.reduce_sum(v_memory_vectors * query_vector, axis=-1)
        weighted_mask = tf.nn.softmax(tf.reduce_sum(memory_vectors * query_vector, axis=-1))
        weighted_mask *= utter_num_mask
        return weighted_mask

    def self_attention(self, memory_vectors, hidden_size, scope=None, reuse=None):
        with tf.variable_scope(scope or "self_attention", reuse=reuse):
            memory_vectors = tf.expand_dims(memory_vectors, axis=2)
            context_vecotor = tf.get_variable("context_vecotor", initializer=self.glorot_uniform((self.hidden_size_2, 1)))
            context_vecotor = tf.expand_dims(tf.expand_dims(context_vecotor, 0), 0)
            e = tf.reduce_sum(tf.reduce_sum(tf.nn.conv2d(memory_vectors, context_vecotor, [1, 1, 1, 1], "SAME"), 2), -1)
            def masked_attention(e):
                attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
            attn_dist = masked_attention(e)
            return attn_dist

    def static_weighted(self, memory_vectors):
        # W_t = tf.get_variable("static_weight", shape=[1, self.input_utter_num, 1])
        W_t = tf.Variable(np.ones((self.utter_len,))[None, :, None], name='static_weight', dtype=tf.float32)
        weighted_vector = tf.reduce_sum(memory_vectors * W_t, -1)
        return weighted_vector

    def add_decoder(self, initial_state, encoder_states, response_last_hidden, cell):
        with tf.variable_scope("attention_decoder") as scope:
            attn_size = encoder_states.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined
            encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)
            attention_vec_size = attn_size
            W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
            encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)
            v = tf.get_variable("v", [attention_vec_size])

            def attention(decoder_state, response_last_hidden):
                with tf.variable_scope("Attention"):
                    with tf.variable_scope("decoder_features"):
                        decoder_features = tf_utils.linear(decoder_state, attention_vec_size, True)  # shape (batch_size, attention_vec_size)
                        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)  # reshape to (batch_size, 1, 1, attention_vec_size)
                    with tf.variable_scope("response_features"):
                        response_features = tf_utils.linear(response_last_hidden, attention_vec_size, True)
                        response_features = tf.expand_dims(tf.expand_dims(response_features, 1), 1)
                    def masked_attention(e):
                        attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
                        masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                        return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
                    e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features + response_features), [2, 3])  # calculate e
                    attn_dist = masked_attention(e)
                    context_vector = tf.reduce_sum(tf.reshape(attn_dist, [-1, self.config.max_utter_len, 1, 1]) * encoder_states, [1, 2])  # shape (batch_size, attn_size).
                    context_vector = tf.reshape(context_vector, [-1, attn_size])
                return context_vector, attn_dist
            state = initial_state
            context_vector, _ = attention(state, response_last_hidden)
            steps = 0
            while steps < self.attention_num:
                cell_output, state = cell(context_vector, state)
                tf.get_variable_scope().reuse_variables()
                context_vector, attn_dist = attention(state, response_last_hidden)
                steps += 1
            return state

    @staticmethod
    def glorot_uniform(size):
        fan_in, fan_out = size
        s = np.sqrt(6. / (fan_in + fan_out))
        return np.random.uniform(size=size,low=-s, high=s).astype(np.float32)

    @staticmethod
    def ortho_weight(ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')
