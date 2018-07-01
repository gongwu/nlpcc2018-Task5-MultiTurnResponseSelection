# -*- coding:utf-8 _*-
import numpy as np
import math
from utils import data_utils
from utils import utils


class Feature(object):
    def __init__(self, config):
        self.vocab = data_utils.load_params(config.w2i_file)
        self.char_dict = data_utils.build_character(self.vocab)
        self.idf_dic = data_utils.load_params(config.idf_file)
        self.word_embed_list = data_utils.load_params(config.we_file)
        self.word_embed_dict = data_utils.load_embed_from_vocab(self.vocab, self.word_embed_list)

    def uni_represent_word(self, post, response):
        """represent unigram to a simple list"""
        post_list = []
        response_list = []
        for word in post:
            if word in self.vocab:
                index = self.vocab[word]
                post_list.append(index)
        for word in response:
            if word in self.vocab:
                index = self.vocab[word]
                response_list.append(index)
        return post_list, response_list

    def uni_represent_char(self, post, response):
        """represent unigram to a simple list"""
        post_list = []
        response_list = []
        for word in post:
            if word in self.char_dict:
                index = self.char_dict[word]
                post_list.append(index)
        for word in response:
            if word in self.char_dict:
                index = self.char_dict[word]
                response_list.append(index)
        return post_list, response_list

    def tfidf_represent_word(self, post, response):
        post_tfidf_dict = {}
        response_tfidf_dict = {}
        for word in post:
            if word not in self.idf_dic:
                idf = 0
            else:
                idf = self.idf_dic[word]
            tf = post.count(word) / float(len(post))
            post_tfidf_dict[word] = tf * idf
        for word in response:
            if word not in self.idf_dic:
                idf = 0
            else:
                idf = self.idf_dic[word]
            tf = response.count(word) / float(len(response))
            response_tfidf_dict[word] = tf * idf
        return post_tfidf_dict, response_tfidf_dict

    # 构造两个句子间的词匹配 7D Feature
    def word_match(self, post, response):
        words1 = set(post)
        words2 = set(response)
        words1_len = len(words1)
        words2_len = len(words2)
        if not words1_len or not words2_len:
            return list([0] * 7)
        union_num = len(words1 | words2)
        intersection_num = len(words1 & words2)
        w1_w2_num = len(words1 - words2)
        w2_w1_num = len(words2 - words1)
        feat1 = float(intersection_num)
        feat2 = float(intersection_num) / words1_len
        feat3 = float(intersection_num) / words2_len
        feat4 = float(w2_w1_num) / words2_len
        feat5 = float(w1_w2_num) / words1_len
        feat6 = float(intersection_num) / union_num
        feat7 = float(union_num - intersection_num) / union_num
        feat = [feat1, feat2, feat3, feat4, feat5, feat6, feat7]
        return list(feat)

    # 构造两个句子间的字符匹配
    def char_match(self, post, response):
        diff_len = abs(len(post) - len(response))
        post = set(post)
        response = set(response)
        post_len = len(post)
        response_len = len(response)
        if not post_len or not response_len:
            return list([0] * 5)
        union_num = len(post | response)
        intersection_num = len(post & response)
        w1_w2_num = len(post - response)
        w2_w1_num = len(response - post)
        feat1 = float(intersection_num)
        feat2 = float(intersection_num) / post_len
        feat3 = float(intersection_num) / response_len
        feat4 = float(w2_w1_num) / response_len
        feat5 = float(w1_w2_num) / post_len
        feat6 = float(intersection_num) / union_num
        feat7 = float(union_num - intersection_num) / union_num
        feat = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, diff_len]
        return list(feat)

    # 余弦距离
    def cosine(self, vector1, vector2):
        a = (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if a == 0:
            return 1
        cosV12 = np.dot(vector1, vector2) / a
        return cosV12

    def tfidf_Cos_distance(self, dict1, dict2):
        mem_sum = 0  # 分子和
        den_sum = 0  # 分母和
        den_sum1 = den_sum2 = 0
        for dim in dict1:
            if dim in dict2:
                mem_sum += dict1[dim] * dict2[dim]
            den_sum1 += dict1[dim] ** 2
        for dim in dict2:
            den_sum2 += dict2[dim] ** 2
        den_sum3 = math.sqrt(den_sum1)
        den_sum4 = math.sqrt(den_sum2)
        den_sum = den_sum3 + den_sum4
        if den_sum == 0:
            re = 0
        else:
            re = mem_sum / den_sum
        return re

    # 基于unigram的余弦距离
    def uni_Cos_distance(self, vec1, vec2):
        """calculate the cos distance of two sample list """
        mem_sum = 0  # 分子和
        den_sum = 0  # 分母和
        for dim in vec1:
            if dim in vec2:
                mem_sum += 1
        den_sum1 = math.sqrt(len(vec1))
        den_sum2 = math.sqrt(len(vec2))
        den_sum = den_sum1 + den_sum2
        if den_sum == 0:
            return -1
        return mem_sum / den_sum

    # 基于unigram的曼哈顿距离
    def uni_Man_distance(self, vec1, vec2):
        """calculate the Manhattan distance of two sample list"""
        coin_sum = 0
        len_vec1 = len(vec1)
        len_vec2 = len(vec2)
        for dim in vec1:
            if dim in vec2:
                coin_sum += 1
        man = len_vec1 + len_vec2 - coin_sum
        return man

    # 基于unigram的杰卡德距离
    def uni_Jaccard_distance(self, vec1, vec2):
        post = set(vec1)
        response = set(vec2)
        union = len(post | response)
        intersection = len(post & response)
        if union == 0:
            return -1
        jac_distance = 1 - (intersection / union)
        return jac_distance

    # 计算所有基于unigram的距离
    def unigram_distance(self, post, response):
        """build 3 distances for unigram on word an char level"""
        post_word, response_word = self.uni_represent_word(post, response)
        post_list_char, response_list_char = self.make_words2char(post, response)
        post_char, response_char = self.uni_represent_char(post_list_char, response_list_char)
        post_tfidf_word, response_tfidf_word = self.tfidf_represent_word(post, response)
        cos_tfidf_word = self.tfidf_Cos_distance(post_tfidf_word, response_tfidf_word)
        cos_word = self.uni_Cos_distance(post_word, response_word)
        man_word = self.uni_Man_distance(post_word, response_word)
        jac_word = self.uni_Jaccard_distance(post_word, response_word)
        cos_char = self.uni_Cos_distance(post_char, response_char)
        man_char = self.uni_Man_distance(post_char, response_char)
        jac_char = self.uni_Jaccard_distance(post_char, response_char)
        return list([cos_word, man_word, jac_word, cos_char, man_char, jac_char, cos_tfidf_word])

    # 构造字符级别的句子表示
    def make_words2char(self, post, response):
        post_str = "".join(post)
        post_list = list(post_str)
        response_str = "".join(response)
        response_list = list(response_str)
        return post_list, response_list

    # 将字符匹配特征存储到特征类
    def words2char_match(self, post, response):
        """build char_sent match to feature"""
        post_list, response_list = self.make_words2char(post, response)
        fea_chars = self.char_match(post_list, response_list)
        return fea_chars

    # 基于词向量的切比雪夫距离
    def Chebyshev_Distance(self, vector1, vector2):
        return abs(vector1 - vector2).max()

    # 基于词向量的曼哈顿距离
    def Manhattan_Distance(self, vector1, vector2):
        return sum(abs(vector1 - vector2))

    # 基于词向量的欧式距离
    def Euclidean_Distance(self, vector1, vector2):
        return np.sqrt(sum(np.square(vector1 - vector2)))

    def diff_vecs(self, vec1, vec2):
        if not len(vec1) == len(vec2):
            print("Error: dim disagrees")
        vec = vec1[:]
        for index in range(len(vec)):
            vec[index] -= vec2[index]
        return vec

    def Manhattan_dist(self, vec1, vec2):
        vec = self.diff_vecs(vec1, vec2)
        sum = 0.0
        for val in vec:
            sum += abs(val)
        return sum

    def Euclidean_dist(self, vec1, vec2):
        vec = self.diff_vecs(vec1, vec2)
        sum = 0.0
        for val in vec:
            sum += val * val
        return math.sqrt(sum)

    def polynomial(self, v1, v2, degree=3, gamma=None, coef0=1):
        """
        K(X, Y) = (gamma <X, Y> + coef0)^degree
        :param v1: numpy vector
        :param v2:
        :return:
        """
        if gamma is None:
            gamma = 1.0 / v1.shape[0]
        K = np.dot(v1, v2)
        K *= gamma
        K += coef0
        K **= degree
        return K

    def rbf(self, v1, v2, gamma=None):
        """
         K(x, y) = exp(-gamma ||x-y||^2)
        :param v1:
        :param v2:
        :param gamma:
        :return:
        """
        if gamma is None:
            gamma = 1.0 / v1.shape[0]
        K = self.Euclidean_Distance(v1, v2)
        K *= -gamma
        K = np.exp(K)
        return K

    def laplacian(self, v1, v2, gamma=None):
        """
         K(x, y) = exp(-gamma ||x-y||_1)
        :param v1:
        :param v2:
        :return:
        """
        if gamma is None:
            gamma = 1.0 / v1.shape[0]
        K = self.Manhattan_Distance(v1, v2)
        K *= -gamma
        K = np.exp(K)
        return K

    def sigmoid(self, v1, v2, gamma=None, coef0=1):
        """
        K(X, Y) = tanh(gamma <X, Y> + coef0)
        :param v1:
        :param v2:
        :return:
        """
        if gamma is None:
            gamma = 1.0 / v1.shape[0]
        K = np.dot(v1, v2)
        K *= gamma
        K += coef0
        K = np.tanh(K)  # compute tanh in-place
        return K

    # 获取基于词向量的距离特征
    def get_word_embedding_distance(self, post, response):
        aver_vec1, aver_vec2 = self.aver_vector2sent(post, response, 200)
        max_vec1, max_vec2 = self.max_vector2sent(post, response, 200)
        min_vec1, min_vec2 = self.min_vector2sent(post, response, 200)
        Non_Lin_Ker = [self.sigmoid, self.laplacian, self.polynomial, self.rbf]
        non_liner_kel = []
        for kernel in Non_Lin_Ker:
            aver = kernel(aver_vec1, aver_vec2)
            max = kernel(max_vec1, max_vec2)
            min = kernel(min_vec1, min_vec2)
            non_liner_kel.extend([aver, max, min])
        cos_aver = self.cosine(aver_vec1, aver_vec2)
        che_aver = self.Chebyshev_Distance(aver_vec1, aver_vec2)
        man_aver = self.Manhattan_Distance(aver_vec1, aver_vec2)
        euc_aver = self.Euclidean_Distance(aver_vec1, aver_vec2)
        cos_max = self.cosine(max_vec1, max_vec2)
        che_max = self.Chebyshev_Distance(max_vec1, max_vec2)
        man_max = self.Manhattan_Distance(max_vec1, max_vec2)
        euc_max = self.Euclidean_Distance(max_vec1, max_vec2)
        cos_min = self.cosine(min_vec1, min_vec2)
        che_min = self.Chebyshev_Distance(min_vec1, min_vec2)
        man_min = self.Manhattan_Distance(min_vec1, min_vec2)
        euc_min = self.Euclidean_Distance(min_vec1, min_vec2)
        liner_kel = [cos_aver, cos_max, cos_min, man_aver, man_max, man_min, euc_aver, euc_max, euc_min, che_aver, che_max,
                     che_min]
        return list(liner_kel + non_liner_kel)

    # 句子的平均向量表示
    def aver_vector2sent(self, post, response, dim):
        post_embed = response_embed = np.zeros(dim)
        for word in post:
            if word in self.word_embed_dict:
                v1 = self.word_embed_dict[word]
            else:
                v1 = np.random.uniform(-0.5, 0.5, dim)
            post_embed += v1
        for word in response:
            if word in self.word_embed_dict:
                v2 = self.word_embed_dict[word]
            else:
                v2 = np.random.uniform(-0.5, 0.5, dim)
            response_embed += v2
        return post_embed / len(post), response_embed / len(response)

    # 句子的最大向量表示
    def max_vector2sent(self, post, response, dim):
        post_embed = response_embed = []
        for word in post:
            if word in self.word_embed_dict:
                v1 = self.word_embed_dict[word]
            else:
                v1 = np.random.uniform(-0.5, 0.5, dim)
            post_embed.append(v1)
        post_max_embed = np.max(post_embed, axis=0)
        for word in response:
            if word in self.word_embed_dict:
                v2 = self.word_embed_dict[word]
            else:
                v2 = np.random.uniform(-0.5, 0.5, dim)
            response_embed.append(v2)
        response_max_embed = np.max(response_embed, axis=0)
        return post_max_embed, response_max_embed

    # 句子的最小向量表示
    def min_vector2sent(self, post, response, dim):
        post_embed = response_embed = []
        for word in post:
            if word in self.word_embed_dict:
                v1 = self.word_embed_dict[word]
            else:
                v1 = np.random.uniform(-0.5, 0.5, dim)
            post_embed.append(v1)
        post_min_embed = np.min(post_embed, axis=0)
        for word in response:
            if word in self.word_embed_dict:
                v2 = self.word_embed_dict[word]
            else:
                v2 = np.random.uniform(-0.5, 0.5, dim)
            response_embed.append(v2)
        response_min_embed = np.min(response_embed, axis=0)
        return post_min_embed, response_min_embed

    def tf_idf(self, word, count, count_list):
        tf = count[word] / sum(count.values())
        n_contain = sum(1 for count in count_list if word in count)
        idf = math.log(len(count_list) / (1 + n_contain(word, count_list)))
        return tf * idf

    def __call__(self, utter, response):
        feature_function_list = [self.word_match, self.words2char_match, self.unigram_distance]
        features = [feature_function(utter, response) for feature_function in feature_function_list]
        feature = np.concatenate(features, axis=0)
        return list(utils.normalize(feature))

