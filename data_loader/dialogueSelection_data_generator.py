# -*- coding:utf-8 -*-
import numpy as np
from base.data_generator import DataGenerator
from utils import data_utils
from feature.make_feature import Feature


class DialogueDataGenerator(DataGenerator):
    def __init__(self, config):
        super(DialogueDataGenerator, self).__init__(config)
        self.max_sent_len = self.config.max_sent_len
        self.max_utter_len = self.config.max_utter_len
        self.train_file = self.config.train_file
        self.dev_file = self.config.dev_file
        self.test_file = self.config.test_file
        self.init_embedding()

    def build_data(self, data_file):
        self.examples = data_utils.read_dialogue_selection_data(data_file)
        feature = Feature(self.config)
        y = []
        f_utters = []
        f_response = []
        f_u_feature = []
        f_p_feature = []
        response_lens = []
        utters_lens = []
        utters_num = []
        for example in self.examples:
            utter = example[0]
            response = example[1]
            label = int(example[2])
            if self.config.feature_p:
                post = list(np.concatenate(utter, axis=0))
                fea_p = feature(post, response)
                f_p_feature.append(fea_p)
            if self.config.feature_u:
                u_feature = []
                for u in utter:
                    fea_u = feature(u, response)
                    u_feature.append(fea_u)
                f_u_feature.append(u_feature)
            # 这里一定要用vocab,因为utter_vocab已经发生了变化
            utters = data_utils.char_to_matrix(utter, self.vocab)
            responses = data_utils.sent_to_index(response, self.res_vocab)
            one_hot_label = data_utils.onehot_vectorize(label, self.config.num_class)
            y.append(one_hot_label)
            f_utters.append(utters)
            f_response.append(responses)
            response_lens.append(min(len(response), self.max_sent_len))
            utters_num.append(min(len(utter), self.max_utter_len))
            utters_lens.append([min(len(sent), self.max_sent_len) for sent in utter])

        input_x_utter = data_utils.pad_3d_tensor(f_utters, self.max_utter_len, self.max_sent_len)
        input_x_response = data_utils.pad_2d_matrix(f_response, self.max_sent_len)
        utters_lens = data_utils.pad_2d_matrix(utters_lens, self.max_utter_len)
        utter_weighted_mask = data_utils.pad_2d_weighted_mask(utters_num, self.max_utter_len)
        utter_padding_mask = data_utils.pad_2d_mask(utters_num, self.max_utter_len)
        utter_final_padding_mask = data_utils.pad_2d_one_mask(utters_num, self.max_utter_len)
        if self.config.feature_u:
            f_u_features = data_utils.pad_3d_tensor(f_u_feature, self.max_utter_len, 22)
        x_res_len = response_lens
        x_utter_len = utters_lens
        self.input_x_response = np.array(input_x_response, dtype=np.int32)  # [batch_size, sent_len]
        self.input_x_utter = np.array(input_x_utter, dtype=np.int32)
        if self.config.feature_p:
            self.input_x_feature_p = np.array(f_p_feature, dtype=np.float32)
        if self.config.feature_u:
            self.input_x_feature_u = np.array(f_u_features, dtype=np.float32)
        self.x_res_len = np.array(x_res_len, dtype=np.int32)  # [batch_size]
        self.x_utter_num = np.array(utters_num, dtype=np.int32)  # [batch_size]
        self.x_utter_weighted_mask = np.array(utter_weighted_mask, dtype=np.float32)  # [batch_size, utter_len]
        self.x_utter_mask = np.array(utter_padding_mask, dtype=np.int32)  # [batch_size, utter_len]
        self.x_utter_final_mask = np.array(utter_final_padding_mask, dtype=np.int32)  # [batch_size, utter_len]
        self.x_utter_len = np.array(x_utter_len, dtype=np.int32)
        self.y = np.array(y, dtype=np.float32)  # [batch_size, class_number]

    def next_batch(self, batch_size, shuffle=False):
        input_x_response = self.input_x_response
        input_x_utter = self.input_x_utter
        if self.config.feature_p:
            input_x_feature_p = self.input_x_feature_p
        if self.config.feature_u:
            input_x_feature_u = self.input_x_feature_u
        x_res_len = self.x_res_len
        x_utter_num = self.x_utter_num
        x_utter_len = self.x_utter_len
        x_utter_weighted_mask = self.x_utter_weighted_mask
        x_utter_mask = self.x_utter_mask
        x_utter_final_mask = self.x_utter_final_mask
        y = self.y
        assert len(input_x_response) == len(y)
        assert len(input_x_utter) == len(y)
        n_data = len(y)
        idx = np.arange(n_data)
        if shuffle:
            idx = np.random.permutation(n_data)
        for start_idx in range(0, n_data, batch_size):
            # end_idx = min(start_idx + batch_size, n_data)
            end_idx = start_idx + batch_size
            excerpt = idx[start_idx:end_idx]
            batch = data_utils.Batch()
            batch.add('res', input_x_response[excerpt])
            batch.add('utter', input_x_utter[excerpt])
            batch.add('utter_num', x_utter_num[excerpt])
            batch.add('enc_padding_mask', x_utter_mask[excerpt])
            batch.add('final_padding_mask', x_utter_final_mask[excerpt])
            batch.add('utter_weighted_mask', x_utter_weighted_mask[excerpt])
            batch.add('res_len', x_res_len[excerpt])
            batch.add('utter_len', x_utter_len[excerpt])
            batch.add('label', y[excerpt])
            if self.config.feature_p:
                batch.add('feature_p', input_x_feature_p[excerpt])
            if self.config.feature_u:
                batch.add('feature_u', input_x_feature_u[excerpt])
            yield batch

    def build_vocab(self):
        if self.test_file is None:
            print('test_file is None')
            file_list = [self.train_file, self.dev_file]
        else:
            file_list = [self.train_file, self.dev_file, self.test_file]
        examples = data_utils.read_dialogue_selection_data(file_list)
        utters = []
        responses = []
        for example in examples:
            utter = example[0]
            response = example[1]
            utters.append(utter)
            responses.append(response)
        res_vocab = data_utils.build_word_vocab(responses, self.threshold)
        utter_vocab = data_utils.build_utter_vocab(utters, self.threshold)
        # 这里必须将vocab_res放在vocab_utter的后面，不然在decoder阶段会出现index不对应
        vocab = data_utils.merge_vocab(res_vocab, utter_vocab)
        # 统计平均长度与最大长度
        max_sent_len = 0
        avg_sent_len = 0
        for sent in responses:
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            avg_sent_len += len(sent)
        avg_sent_len /= len(responses)
        print('task: max_sent_len: {}'.format(max_sent_len))
        print('task: avg_sent_len: {}'.format(avg_sent_len))
        max_utters_num = 0
        avg_utters_num = 0
        for utter in utters:
            if len(utter) > max_utters_num:
                max_utters_num = len(utter)
            avg_utters_num += len(utter)
        avg_utters_num /= len(utters)
        print('task: max_utters_num: {}'.format(max_utters_num))
        print('task: avg_utters_num: {}'.format(avg_utters_num))
        return utter_vocab, res_vocab, vocab

    def init_embedding(self):
        self.word_embed_file = self.config.word_embed_file
        self.word_dim = self.config.word_dim
        self.threshold = self.config.threshold
        self.we_file = self.config.we_file
        self.w2i_file = self.config.w2i_file
        self.r2i_file = self.config.r2i_file
        self.u2i_file = self.config.u2i_file

        # the char_embed always init
        if self.config.init:
            self.utter_vocab, self.res_vocab, self.vocab = self.build_vocab()
            self.embed = data_utils.load_word_embedding(self.vocab, self.word_embed_file, self.config, self.word_dim)
            data_utils.save_params(self.vocab, self.w2i_file)
            data_utils.save_params(self.res_vocab, self.r2i_file)
            data_utils.save_params(self.utter_vocab, self.u2i_file)
            data_utils.save_params(self.embed, self.we_file)
        else:
            self.embed = data_utils.load_params(self.we_file)
            self.vocab = data_utils.load_params(self.w2i_file)
            self.res_vocab = data_utils.load_params(self.r2i_file)
            self.utter_vocab = data_utils.load_params(self.u2i_file)
            self.embed = self.embed.astype(np.float32)
        print("vocab size: %d" % len(self.vocab), "we shape: ", self.embed.shape)