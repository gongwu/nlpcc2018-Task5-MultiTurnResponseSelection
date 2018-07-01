# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest


def fwLSTM(cell_fw, input_x, input_x_len):
    outputs_fw, states_fw = tf.nn.dynamic_rnn(cell=cell_fw, inputs=input_x, sequence_length=input_x_len,
                                              dtype=tf.float32, scope='fw')
    return outputs_fw


def bwLSTM(cell_bw, input_x, input_x_len, time_major=False):
    """
    Ref: bidirectional dynamic rnn in tensorflow r1.0
    https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/python/ops/rnn.py#L255-L377
    """
    if not time_major:
        time_dim = 1
        batch_dim = 0
    else:
        time_dim = 0
        batch_dim = 1

    inputs_reverse = array_ops.reverse_sequence(input=input_x, seq_lengths=input_x_len, seq_dim=time_dim,
            batch_dim=batch_dim)
    tmp, output_state_bw = tf.nn.dynamic_rnn(cell=cell_bw, inputs=inputs_reverse, sequence_length=input_x_len,
            dtype=tf.float32, scope='bw')

    outputs_bw = array_ops.reverse_sequence(input=tmp, seq_lengths=input_x_len, seq_dim=time_dim, batch_dim=batch_dim)

    return outputs_bw


def BiLSTM(lstm_size, input_x, input_x_len, dropout_keep_rate):
    cell_fw = tf.contrib.rnn.GRUCell(lstm_size)
    cell_bw = tf.contrib.rnn.GRUCell(lstm_size)

    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_rate)

    # b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
    #                                                       sequence_length=input_x_len, dtype=tf.float32)

    outputs_fw = fwLSTM(cell_fw, input_x, input_x_len)
    outputs_bw = bwLSTM(cell_bw, input_x, input_x_len)
    return outputs_fw, outputs_bw


def AvgPooling(input_x, input_len, max_input_len):
    """
    Avg_Pooling
    Args:
        input_x: [batch, sent, embedding]
        input_len: [batch]
    Returns:
        [batch, sent_embedding]
    """

    mask = tf.sequence_mask(input_len, max_input_len, dtype=tf.float32)
    norm = mask / (tf.reduce_sum(mask, -1, keep_dims=True) + 1e-30)
    output = tf.reduce_sum(input_x * tf.expand_dims(norm, -1), axis=1)
    return output


def mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = mask_and_avg(covlosses, padding_mask)
    return coverage_loss


def Mask(input_x, input_len, max_input_len):
    mask = tf.sequence_mask(input_len, max_input_len, dtype=tf.float32)
    input_mask = input_x * tf.expand_dims(mask, -1)
    return input_mask


def MaxPooling(input_x, input_lengths):
    """
    Max pooling.
    Args:
        input_x: [batch, max_sent_len, embedding]
        input_lengths: [batch]
    Returns:
        [batch, sent_embedding]
    """
    max_sent_len = tf.shape(input_x)[1]
    mask = tf.sequence_mask(input_lengths, max_sent_len, dtype=tf.float32)
    mask = tf.expand_dims((1 - mask) * -1e30, -1)
    output = tf.reduce_max(input_x + mask, axis=1)

    return output


def CNN_Pooling(inputs, filter_sizes=(1, 2, 3, 5), num_filters=100):
    """
    CNN-MaxPooling
    inputs: [batch_size, sequence_length, hidden_size]
    filter_sizes: list, [1, 2, 3, 5]
    num_filters: int, 500
    :return:
    """
    sequence_length = inputs.get_shape()[1]
    input_size = inputs.get_shape()[2]
    inputs = tf.expand_dims(inputs, axis=-1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, input_size, 1, num_filters]
            W = tf.get_variable("W", filter_shape, initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='VALID', name="conv-1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                    padding='VALID', name="poll-1")
            pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    pooled_reshape = tf.reshape(tf.concat(pooled_outputs, axis=3), [-1, num_filters_total])
    return pooled_reshape


def dot_product_attention(question_rep, passage_repres, passage_mask):
    """
    Attention dot_product
      Args:
        question_rep: [batch_size, hidden_size]
        passage_repres: [batch_size, sequence_length, hidden_size]
        passage_mask: [batch_size, sequence_length]
      Returns:
        passage_rep: [batch_size, hidden_size]
    """
    question_rep = tf.expand_dims(question_rep, 1)
    passage_prob = softmask(tf.reduce_sum(question_rep * passage_repres, axis=2), passage_mask)
    passage_rep = tf.reduce_sum(passage_repres * tf.expand_dims(passage_prob, axis=-1), axis=1)
    return passage_rep


def bilinear_attention(question_rep, passage_repres, passage_mask):
    """
    Attention bilinear
    adopt from danqi, https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py
      Args:
        question_rep: [batch_size, hidden_size]
        passage_repres: [batch_size, sequence_length, hidden_size]
        passage_mask: [batch_size, sequence_length]
      Returns:
        passage_rep: [batch_size, hidden_size]
    """
    hidden_size = question_rep.get_shape()[1]
    W_bilinear = tf.get_variable("W_bilinear", shape=[hidden_size, hidden_size], dtype=tf.float32)

    question_rep = tf.matmul(question_rep, W_bilinear)
    question_rep = tf.expand_dims(question_rep, 1)
    alpha = tf.nn.softmax(tf.reduce_sum(question_rep * passage_repres, axis=2))
    alpha = alpha * passage_mask
    alpha = alpha / tf.reduce_sum(alpha, axis=-1, keep_dims=True)

    passage_rep = tf.reduce_sum(passage_repres * tf.expand_dims(alpha, axis=-1), axis=1)
    return passage_rep


def softmask(input_prob, input_mask, eps=1e-6):
    """
    normarlize the probability
    :param input_prob: [batch_size, sequence_length]
    :param input_mask: [batch_size, sequence_length]
    :return: [batch_size, sequence]
    """
    input_prob = tf.exp(input_prob, name='exp')
    input_prob = input_prob * input_mask
    input_sum = tf.reduce_sum(input_prob, axis=1, keep_dims=True)
    input_prob = input_prob / (input_sum + eps)
    return input_prob


def length(data):
    """
    calculate length, according to zero
    :param data:
    :return:
    """
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    length_one = tf.ones(tf.shape(length), dtype=tf.int32)
    length = tf.maximum(length, length_one)
    return length


def last_relevant(output, length):
    """
    fetch the last relevant
    """
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def context_representation(question_repres, question_len, passage_repres, passage_len, lstm_size, dropout_rate,
                           name_scope=None, reuse=None):
    """

    :param name_scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name_scope or 'BiLSTM_Layer', reuse=reuse):
        # [batch, time_step, n_hidden]

        cell_fw = tf.contrib.rnn.GRUCell(lstm_size)
        cell_bw = tf.contrib.rnn.GRUCell(lstm_size)

        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - dropout_rate))
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - dropout_rate))

        # stack lstm : tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)

        # passage representation : [batch_size, passage_len, context_lstm_dim]
        question_b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, question_repres, question_len,
                                                                       dtype=tf.float32)

        tf.get_variable_scope().reuse_variables()
        passage_b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, passage_repres, passage_len,
                                                                      dtype=tf.float32)

    # b_outputs: tuple: (context_repres_fw, context_repres_bw)
    return question_b_outputs, passage_b_outputs


def gate_mechanism(word_repres, lstm_repres, output_size, scope=None, reuse=None):
    # word_repres: [batch_size, passage_len, dim]
    input_shape = tf.shape(word_repres)
    batch_size = input_shape[0]
    passage_len = input_shape[1]

    word_repres = tf.reshape(word_repres, [batch_size * passage_len, output_size])
    lstm_repres = tf.reshape(lstm_repres, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "gate_layer", reuse=reuse):
        gate_word_w = tf.get_variable("gate_word_w", [output_size, output_size], dtype=tf.float32)
        gate_lstm_w = tf.get_variable("gate_lstm_w", [output_size, output_size], dtype=tf.float32)

        gate_b = tf.get_variable("gate_b", [output_size], dtype=tf.float32)

        gate = tf.nn.sigmoid(tf.matmul(word_repres, gate_word_w) + tf.matmul(lstm_repres, gate_lstm_w) + gate_b)

        outputs = word_repres * gate + lstm_repres * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def kl_loss(preds, golds):
    golds = tf.maximum(1e-6, golds)
    print(golds)
    preds = tf.maximum(1e-6, preds)
    print(preds)
    loss = golds * (tf.log(golds) - tf.log(preds))
    loss = tf.reduce_sum(loss, axis=-1)
    return loss


def predict_to_score(predicts, num_class):
    """
    Checked: the last is for 0
    ===
    Example:    score=1.2, num_class=3 (for 0-2)
                (0.8, 0.2, 0.0) * (1, 2, 0)
    :param predicts:
    :param num_class:
    :return:
    """
    scores = 0.
    i = 0
    while i < num_class:
        scores += i * predicts[:, i - 1]
        i += 1
    return scores


def optimize(loss, optimize_type, lambda_l2, learning_rate, clipper=50):
    if optimize_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        loss = loss + lambda_l2 * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
    elif optimize_type == 'sgd':
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        min_lr = 0.000001
        _lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, global_step, 30000, 0.98))
        train_op = tf.train.GradientDescentOptimizer(learning_rate=_lr_rate).minimize(loss)
    elif optimize_type == 'ema':
        tvars = tf.trainable_variables()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
        maintain_averages_op = ema.apply(tvars)
        # Create an op that will update the moving averages after each training
        # step.  This is what we will use in place of the usual training op.
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)
    elif optimize_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        loss = loss + lambda_l2 * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(list(zip(grads, tvars)))

    extra_train_ops = []
    train_ops = [train_op] + extra_train_ops
    train_op = tf.group(*train_ops)
    return train_op


def bilinear_attention_layer(question_repres, question_mask, question_rep,
                             passage_repres, passage_mask, passage_rep,
                             out_size=300, scope=None, reuse=None):
    """
    question_repres: [batch_size, sent_length, dim]
    question_mask  : [batch_size, sent_length]
    question_rep   : [batch_size, dim]
    """
    question_mask = tf.cast(question_mask, tf.float32)
    passage_mask = tf.cast(passage_mask, tf.float32)

    with tf.variable_scope(scope or "attention", reuse=reuse):
        W_bilinear = tf.get_variable("W_bilinear", [out_size, out_size], dtype=tf.float32)
        # W_bilinear_2 = tf.get_variable("W_bilinear_2", [out_size, out_size], dtype=tf.float32)

        question_rep = tf.matmul(question_rep, W_bilinear)
        question_rep = tf.expand_dims(question_rep, 1)

        passage_prob = tf.nn.softmax(tf.reduce_sum(passage_repres * question_rep, 2))
        passage_prob = passage_prob * passage_mask / tf.reduce_sum(passage_mask, -1, keep_dims=True)
        passage_outputs = passage_repres * tf.expand_dims(passage_prob, -1)

        passage_rep = tf.matmul(passage_rep, W_bilinear)
        passage_rep = tf.expand_dims(passage_rep, 1)

        question_prob = tf.nn.softmax(tf.reduce_sum(question_repres * passage_rep, 2))
        question_prob = question_prob * question_mask / tf.reduce_sum(question_mask, -1, keep_dims=True)
        question_outputs = question_repres * tf.expand_dims(question_prob, -1)
    return question_outputs, passage_outputs


def alignment_attention(question_repres, passage_repres, passage_align_mask, output_size, scope=None, reuse=None):
    """
    adopt from Qin Chen.
    :param question_repres: [batch_size, sent_len_1, hidden_size]
    :param passage_repres: [batch_size, sent_len_2, hidden_size]
    :param passage_align_mask: [batch_size, sent_len_2, sent_len_1]
    :param output_size: for the variable in gate
    :return: [batch_size, sent_len_2, hidden_size]
    """
    with tf.variable_scope(scope or "align_att", reuse=reuse):
        question_repres = tf.expand_dims(question_repres, 1)
        passage_align_mask = tf.expand_dims(passage_align_mask, -1)
        question_repres = tf.reduce_sum(question_repres * passage_align_mask, axis=1)
        passage_repres = gate_mechanism(question_repres, passage_repres, output_size, scope='align_gate')
    return passage_repres


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                      initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
    """
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
      enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
      cell: rnn_cell.RNNCell defining the cell function and size.
      initial_state_attention:
        Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input.
        If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector.
        If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector.
        We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
      pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
      use_coverage: boolean. If True, use coverage mechanism.
      prev_coverage:
        If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

    Returns:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x cell.output_size]. The output vectors.
      state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
      attn_dists: A list containing tensors of shape (batch_size,attn_length).
        The attention distributions for each decoder step.
      p_gens: List of scalars. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
      coverage: Coverage vector on the last step computed. None if use_coverage=False.
    """
    with tf.variable_scope("attention_decoder") as scope:
        batch_size = encoder_states.get_shape()[0].value  # if this line fails, it's because the batch size isn't defined
        attn_size = encoder_states.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = tf.nn.conv2d(encoder_states, W_h, [1, 1, 1, 1],
                                         "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = tf.get_variable("v", [attention_vec_size])
        if use_coverage:
            with tf.variable_scope("coverage"):
                w_c = tf.get_variable("w_c", [1, 1, 1, attention_vec_size])

        if prev_coverage is not None:  # for beam search mode with coverage
            # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        def attention(decoder_state, coverage=None):
            """Calculate the context vector and attention distribution from the decoder state.

            Args:
              decoder_state: state of the decoder
              coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
              coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
            """
            with tf.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                decoder_features = linear(decoder_state, attention_vec_size,
                                          True)  # shape (batch_size, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),
                                                  1)  # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = tf.nn.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                if use_coverage and coverage is not None:  # non-first step of coverage
                    # Multiply coverage vector by w_c to get coverage_features.
                    coverage_features = tf.nn.conv2d(coverage, w_c, [1, 1, 1, 1],
                                                      "SAME")  # c has shape (batch_size, attn_length, 1, attention_vec_size)

                    # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                    e = tf.reduce_sum(
                        v * tf.tanh(encoder_features + decoder_features + coverage_features),
                        [2, 3])  # shape (batch_size,attn_length)

                    # Calculate attention distribution
                    attn_dist = masked_attention(e)

                    # Update coverage vector
                    coverage += tf.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features),
                                            [2, 3])  # calculate e

                    # Calculate attention distribution
                    attn_dist = masked_attention(e)

                    if use_coverage:  # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = tf.reduce_sum(
                    tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2])  # shape (batch_size, attn_size).
                context_vector = tf.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage  # initialize coverage to None or whatever was passed in
        context_vector = tf.zeros([batch_size, attn_size])
        context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _, coverage = attention(initial_state,
                                                    coverage)  # in decode mode, this is what updates the coverage vector
        for i, inp in enumerate(decoder_inputs):
            tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with tf.variable_scope(tf.get_variable_scope(),
                                                   reuse=True):  # you need this because you've already run the initial attention(...) call
                    context_vector, attn_dist, _ = attention(state, coverage)  # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear([context_vector, state.c, state.h, x], 1, True)  # a scalar
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with tf.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = tf.reshape(coverage, [batch_size, -1])

        return outputs, state, attn_dists, p_gens, coverage


def context_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                 start_of_sequence_id, end_of_sequence_id,
                                 maximum_length, num_decoder_symbols, context_vector,
                                 dtype=tf.int32, name=None, decode_type='greedy'):
    """ Simple decoder function for a sequence-to-sequence model used in the `dynamic_rnn_decoder`.

      Args:
        output_fn: An output function to project your `cell_output` onto class logits.
        If `None` is supplied it will act as an identity function, which might be wanted when using the RNNCell `OutputProjectionWrapper`.
        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        context_vector: an extra vector that should be appended to the input embedding
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
          defaults to "simple_decoder_fn_inference"
      Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
      """
    with tf.name_scope(name, "simple_decoder_fn_inference",
                        [output_fn, encoder_state, embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, num_decoder_symbols, dtype]):
        start_of_sequence_id = tf.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = tf.convert_to_tensor(end_of_sequence_id, dtype)
        maxium_length_int = maximum_length + 1
        maximum_length = tf.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = tf.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = tf.nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = tf.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """ Decoder function used in the `dynamic_rnn_decoder` with the purpose of
            inference.

            The main difference between this decoder function and the `decoder_fn` in
            `simple_decoder_fn_train` is how `next_cell_input` is calculated. In this
            decoder function we calculate the next input by applying an argmax across
            the feature dimension of the output from the decoder. This is a
            greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
            use beam-search instead.

            Args:
              time: positive integer constant reflecting the current timestep.
              cell_state: state of RNNCell.
              cell_input: input provided by `dynamic_rnn_decoder`.
              cell_output: output of RNNCell.
              context_state: context state provided by `dynamic_rnn_decoder`.

            Returns:
              A tuple (done, next state, next input, emit output, next context state)
              where:

              done: A boolean vector to indicate which sentences has reached a
              `end_of_sequence_id`. This is used for early stopping by the
              `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
              all elements as `true` is returned.

              next state: `cell_state`, this decoder function does not modify the
              given state.

              next input: The embedding from argmax of the `cell_output` is used as
              `next_input`.

              emit output: If `output_fn is None` the supplied `cell_output` is
              returned, else the `output_fn` is used to update the `cell_output`
              before calculating `next_input` and returning `cell_output`.

              next context state: `context_state`, this decoder function does not
              modify the given context state. The context state could be modified when
              applying e.g. beam search.
        """
        with tf.name_scope(name, "simple_decoder_fn_inference",
                            [time, cell_state, cell_input, cell_output,
                             context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" % cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = tf.ones([batch_size, ], dtype=dtype) * (
                    start_of_sequence_id)
                done = array_ops.zeros([batch_size, ], dtype=tf.bool)
                cell_state = encoder_state
                cell_output = tf.zeros([num_decoder_symbols],
                                              dtype=tf.float32)
                context_state = tf.zeros((batch_size, maxium_length_int), dtype=tf.int32)
            else:
                cell_output = output_fn(cell_output)

                if decode_type == 'sample':
                    matrix_U = -1.0 * tf.log(
                        -1.0 * tf.log(tf.random_uniform(tf.shape(cell_output), minval=0.0, maxval=1.0)))
                    next_input_id = tf.cast(
                        tf.argmax(tf.subtract(cell_output, matrix_U), dimension=1), dtype=dtype)
                elif decode_type == 'greedy':
                    next_input_id = tf.cast(
                        tf.argmax(cell_output, 1), dtype=dtype)
                else:
                    raise ValueError("unknown decode type")

                done = tf.equal(next_input_id, end_of_sequence_id)
                # save the results into context state
                expanded_next_input = tf.expand_dims(next_input_id, axis=1)
                sliced_context_state = tf.slice(context_state, [0, 0], [-1, maxium_length_int - 1])
                context_state = tf.concat([expanded_next_input, sliced_context_state], axis=1)
                context_state = tf.reshape(context_state, [batch_size, maxium_length_int])

            next_input = tf.gather(embeddings, next_input_id)
            if context_vector is not None:
                next_input = tf.concat([next_input, context_vector], axis=1)
            # if time > maxlen, return all true vector
            done = tf.cond(tf.greater(time, maximum_length),
                                         lambda: tf.ones([batch_size, ], dtype=tf.bool),
                                         lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
