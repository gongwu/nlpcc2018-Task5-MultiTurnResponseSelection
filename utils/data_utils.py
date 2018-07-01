# coding: utf8
import random
import numpy as np
import re
import os
import pickle
import json
import itertools
import codecs

pad_word = '__PAD__'
unk_word = '__UNK__'
start_word = '__START__'
stop_word = '__STOP__'
# set_neg = set([t.strip() for t in open(config.NEGATION_PATH)])
# punc = set([".", ",", "?", "!", "...", ";"])


def read_data(file_list):
    """
    load data from file list
    Args: file_list:
    Returns:
    """
    if type(file_list) != list:
        file_list = [file_list]

    examples = []
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            for line in f:
                items = line.strip().split('\t')
                label = items[0]
                sent = items[1].split()
                examples.append((sent, label))
    return examples


def read_json_data(file_list, segmenter):
    if type(file_list) != list:
        file_list = [file_list]
    examples = []
    for file in file_list:
        tweets = load_json(file)
        for tweet in tweets:
            # add the feature
            sents = get_text_unigram(tweet, segmenter)
            lemmas = get_text_lemmas(tweet, segmenter)
            ners = get_text_ner(tweet)
            pos = get_text_pos(tweet)
            label = tweet["label"]
            id = 0
            # if file == config.train_file:
            #     id = tweet["id"]
            text = tweet["cleaned_text"]
            examples.append((sents, label, ners, pos, lemmas, id, text))
    return examples


def read_dialogue_selection_data(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    examples = []
    for file in file_list:
        dialogues = load_json(file)
        for dialogue in dialogues:
            # add the feature
            utters = dialogue["utterance"]
            response = dialogue["response"]
            label = dialogue["label"]
            examples.append((utters, response, label))
    return examples


def read_dialogue_generator_data(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    examples = []
    for file in file_list:
        dialogues = load_json(file)
        for dialogue in dialogues:
            # add the feature
            utters = dialogue["utterance"]
            response = dialogue["response"]
            examples.append((utters, response))
    return examples


def load_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        tweet_list = json.load(f)
    return tweet_list


def set_dict_key_value(dict, key):
    if key not in dict:
        dict[key] = 0
    dict[key] += 1


def get_text_unigram(microblog, segmenter):
    tokens = microblog["parsed_text"]["tokens"]  # clean_text做预处理得到的分词结果
    ners = microblog["parsed_text"]["ners"]
    pos = microblog["parsed_text"]["pos"]
    wanted_tokens = _process_ngram_tokens(tokens, pos, ners, segmenter)  # 去掉各种number及长度小于2的词
    return list(itertools.chain(*wanted_tokens))


def get_text_lemmas(microblog, segmenter):
    tokens = microblog["parsed_text"]["lemmas"]  # clean_text做预处理得到的分词结果
    ners = microblog["parsed_text"]["ners"]
    pos = microblog["parsed_text"]["pos"]
    wanted_tokens = _process_ngram_tokens(tokens, pos, ners, segmenter)  # 去掉各种number及长度小于2的词
    return list(itertools.chain(*wanted_tokens))


def load_key_value_dict_from_file(dict_file_path):
    dict = {}
    dict_file = open(dict_file_path)
    lines = [line.strip() for line in dict_file]
    dict_file.close()
    for line in lines:
        if line == "":
            continue
        key, value = line.split("\t")
        dict[key] = eval(value)
    return dict


def get_text_ner(microblog):
    ners = microblog["parsed_text"]["ners"]
    return list(itertools.chain(*ners))  # 将多个list拼为1个list


def get_text_pos(microblog):
    poss = microblog["parsed_text"]["pos"]
    return list(itertools.chain(*poss))  # 将多个list拼为1个list


def removeItemsInDict(dict, threshold=1):
    if threshold > 1:
        for key in list(dict.keys()):
            if key == pad_word or key == unk_word or key == start_word or key == stop_word:
                continue
            if dict[key] < threshold:
                dict.pop(key)
    return dict


def _process_ngram_tokens(tokens, pos, ners, segmenter):
    wanted_tokens = []
    for sent_words, sent_pos, sent_ners in zip(tokens, pos, ners):
        wanted_sent_words = []
        for word, pos, ner in zip(sent_words, sent_pos, sent_ners):
            # 去掉各种number
            if ner in ["DATE", "NUMBER", "MONEY", "PERCENT"]:
                # word = ner
                continue
            # if utils.pos2tag(pos) == "#":
            #     continue
            # 将包含数字和单词的token替换成NUMBER_WORD
            if re.search("([0-9]*\.?[0-9]+)", word):
                # word = "NUMBER_WORD"
                continue

            # 去掉hashtag变小写
            # while word.startswith("#"):
            #     word = word[1:].lower()
            # 将hashtag去掉#，加入到句子中
            tag = 0
            while word.startswith("#"):
                word = word[1:].lower()
                tag = 1
            if tag == 1:
                if len(word) >= 2:
                    words = hashtagSegment(segmenter, word)
                    wanted_sent_words += words
                    continue
                else:
                    continue
            # 去掉这些标点符号开头的token
            tag = 0
            punctuations = ["@", "'", ":", ";", "?", "!", "=", "_", "^", "*", "-", ".", "`"]
            for punctuation in punctuations:
                if word.startswith(punctuation):
                # word = word[1:].lower()
                # elif word.endswith(punctuation):
                # word = word[:-1].lower()
                    tag = 1
                    break
            if tag == 1:
                continue
            if word.strip() == "":
                continue
            # 去掉长度小于2的词
            if len(word) < 2:
                continue
            word = word.lower()
            wanted_sent_words.append(word)
        wanted_tokens.append(wanted_sent_words)
    return wanted_tokens


def save_params(params, fname):
    """
    Pickle uses different protocols to convert your data to a binary stream.
    - In python 2 there are 3 different protocols (0, 1, 2) and the default is 0.
    - In python 3 there are 5 different protocols (0, 1, 2, 3, 4) and the default is 3.
    You must specify in python 3 a protocol lower than 3 in order to be able to load
    the data in python 2. You can specify the protocol parameter when invoking pickle.dump.
    """
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=2)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def make_batches(size, batch_size):
    """
    make batch index according to batch_size and size
    :param size: the size of dataset
    :param batch_size: the size of batch
    :return: list: [(0, batch_size), (batch_size, 2*batch_size), ..., (. , min(., .))]
    """
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def vectorize(score, num_class):
    """
    NOT suitable for classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    score = float(score)
    ceil, floor = int(np.ceil(score)), int(np.floor(score))
    if ceil == floor:
        one_hot[floor - 1] = 1
    else:
        one_hot[floor - 1] = ceil - score
        one_hot[ceil - 1] = score - floor
    one_hot = one_hot + 0.00001
    return one_hot


def onehot_vectorize(label, num_class):
    """
    For classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    one_hot[label] = 1.0
    return one_hot


def bow_vectorize(sent_dic, num_vocab):
    bow = np.zeros(num_vocab, dtype=np.float32)
    for idx, value in sent_dic.items():
        bow[idx] = value
    return bow


def sent_to_index(sent, word_vocab):
    """
    :param sent:
    :param word_vocab:
    :return:
    """
    sent_index = []
    for word in sent:
        if word not in word_vocab:
            sent_index.append(word_vocab[unk_word])
        else:
            sent_index.append(word_vocab[word])
    return sent_index


def ner_to_index(ners, ner_vocab):
    """
    :param sent:
    :param ner_vocab:
    :return:
    """
    ner_index = []
    for ner in ners:
        if ner not in ner_vocab:
            ner_index.append(ner_vocab[unk_word])
        else:
            ner_index.append(ner_vocab[ner])
    return ner_index


def pos_to_index(poses, pos_vocab):
    """
    :param sent:
    :param pos_vocab:
    :return:
    """
    pos_index = []
    for pos in poses:
        if pos not in poses:
            pos_index.append(pos_vocab[unk_word])
        else:
            pos_index.append(pos_vocab[pos])
    return pos_index


def rf_to_dict(sents, rf_vocab, word_vocab):
    rf_tweet = {}
    for word in sents:
        if word not in rf_vocab:
            tf = 0
        else:
            tf = rf_vocab[word]
        rf_tweet[word] = tf
    new_feat_dict = {}
    for word in rf_tweet:
        if word in word_vocab:
            new_feat_dict[word_vocab[word]] = rf_tweet[word]
    return new_feat_dict


def rf_to_vector(sents, rf_vocab, word_vocab):
    rf_tweet = {}
    for word in sents:
        if word not in rf_vocab:
            tf = 0
        else:
            tf = rf_vocab[word]
        rf_tweet[word] = tf
    new_feat_dict = np.zeros(len(word_vocab), dtype=np.float32)
    for word in rf_tweet:
        if word in word_vocab:
            new_feat_dict[word_vocab[word]] = rf_tweet[word]
    return new_feat_dict


def get_feature_by_feat_dict(dict, feat_dict):
    new_feat_dict = {}
    for feat in feat_dict:
        if feat in dict:
            new_feat_dict[dict[feat]] = feat_dict[feat]
    return new_feat_dict


def char_to_matrix(sent, char_vocab):
    """
    :param sent
    :param char_vocab
    :return:
    """
    char_matrix = []
    for word in sent:
        char_index = []
        for char in word:
            if char not in char_vocab:
                char_index.append(char_vocab[unk_word])
            else:
                char_index.append(char_vocab[char])
        char_matrix.append(char_index)
    return char_matrix


def pad_1d_vector(words, max_sent_len, dtype=np.int32):
    # 大于最大长度截断， 小于最大长度补0
    padding_words = np.zeros((max_sent_len, ), dtype=dtype)
    kept_length = len(words)
    if kept_length > max_sent_len:
        kept_length = max_sent_len
    padding_words[:kept_length] = words[:kept_length]
    return padding_words


def pad_2d_matrix(batch_words, max_sent_len=None, dtype=np.int32):
    """
    :param batch_words: [batch_size, sent_length]
    :param max_sent_len: if None, max(sent_length)
    :param dtype:
    :return: padding_words: [batch_size, max_sent_length], 0
    """
    if max_sent_len is None:
        max_sent_len = np.max([len(words) for words in batch_words])
    batch_size = len(batch_words)
    padding_words = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        words = batch_words[i]
        kept_length = len(words)
        if kept_length > max_sent_len:
            kept_length = max_sent_len
        padding_words[i, :kept_length] = words[:kept_length]
    return padding_words


def pad_3d_tensor_old(batch_chars, max_sent_length=None, max_word_length=None, dtype=np.int32):
    """
    :param batch_chars: [batch_size, sent_length, word_length]
    :param max_sent_length:
    :param max_word_length:
    :param dtype:
    :return:
    """
    if max_sent_length is None:
        max_sent_length = np.max([len(words) for words in batch_chars])

    if max_word_length is None:
        max_word_length = np.max([np.max([len(chars) for chars in words]) for words in batch_chars])

    batch_size = len(batch_chars)
    padding_chars = np.zeros((batch_size, max_sent_length, max_word_length), dtype=dtype)

    for i in range(batch_size):
        sent_length = max_sent_length
        # 不按最大长度补齐
        if len(batch_chars[i]) < max_sent_length:
            sent_length = len(batch_chars[i])

        for j in range(sent_length):
            chars = batch_chars[i][j]
            kept_length = len(chars)
            if kept_length > max_word_length:
                kept_length = max_word_length
            padding_chars[i, j, :kept_length] = chars[:kept_length]
    return padding_chars


def pad_3d_tensor(batch_chars, max_sent_length=None, max_word_length=None, dtype=np.int32):
    """
    :param batch_chars: [batch_size, sent_length, word_length]
    :param max_sent_length:
    :param max_word_length:
    :param dtype:
    :return:
    """
    if max_sent_length is None:
        max_sent_length = np.max([len(words) for words in batch_chars])

    if max_word_length is None:
        max_word_length = np.max([np.max([len(chars) for chars in words]) for words in batch_chars])

    batch_size = len(batch_chars)
    padding_chars = np.zeros((batch_size, max_sent_length, max_word_length), dtype=dtype)

    for i in range(batch_size):
        sent_length = len(batch_chars[i])
        # 按最后面还是按最前面截断
        if sent_length < max_sent_length:
            for j in range(sent_length):
                chars = batch_chars[i][j]
                kept_length = len(chars)
                if kept_length > max_word_length:
                    kept_length = max_word_length
                padding_chars[i, j, :kept_length] = chars[:kept_length]
        else:
            for j in range(max_sent_length):
                chars = batch_chars[i][j+sent_length-max_sent_length]
                kept_length = len(chars)
                if kept_length > max_word_length:
                    kept_length = max_word_length
                padding_chars[i, j, :kept_length] = chars[:kept_length]
    return padding_chars


def pad_2d_mask(response_lens, max_sent_len=None, dtype=np.int32):
    batch_size = len(response_lens)
    padding_mask = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        enc_len = response_lens[i]
        for j in range(enc_len):
            padding_mask[i][j] = 1
    return padding_mask

def pad_2d_one_mask(response_lens, max_sent_len=None, dtype=np.int32):
    batch_size = len(response_lens)
    padding_one_mask = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        enc_len = response_lens[i]
        padding_one_mask[i][enc_len-1] = 1
    return padding_one_mask

def pad_2d_weighted_mask(response_lens, max_sent_len=None, dtype=np.float32):
    batch_size = len(response_lens)
    padding_mask = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        enc_len = response_lens[i]
        for j in range(enc_len):
            padding_mask[i][j] = 1.0 - ((enc_len - (j + 1) )/ enc_len)
    return padding_mask

def pad_3d_mask(response_lens, max_sent_len=None, dtype=np.int32):
    batch_size = len(response_lens)
    padding_mask = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        enc_len = response_lens[i]
        for j in range(enc_len):
            padding_mask[i][j] = 1
    return padding_mask


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference response as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def get_dec_seqs(sequence, max_len, start_id):
    inp = [start_id] + sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
    return inp


def get_target_seqs(sequence, max_len, stop_id):
    target = sequence[:]
    if len(target) >= max_len:  # truncate
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    return target


def pad_2d_matrix_response(batch_words, word_vocab, max_sent_len=None, dtype=np.int32):
    """
    :param batch_words: [batch_size, sent_length]
    :param max_sent_len: if None, max(sent_length)
    :param dtype:
    :return: padding_words: [batch_size, max_sent_length], 0
    """
    if max_sent_len is None:
        max_sent_len = np.max([len(words) for words in batch_words])
    batch_size = len(batch_words)
    padding_dec_words = np.zeros((batch_size, max_sent_len), dtype=dtype)
    padding_target_words = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        dec_words, target_words = get_dec_inp_targ_seqs(batch_words[i], max_sent_len, word_vocab[start_word], word_vocab[stop_word])
        kept_length = len(dec_words)
        if kept_length > max_sent_len:
            kept_length = max_sent_len
        padding_dec_words[i, :kept_length] = dec_words[:kept_length]
        padding_target_words[i, :kept_length] = target_words[:kept_length]
    return padding_dec_words, padding_target_words


def pad_2d_matrix_response(response_dec, response_target, word_vocab, max_sent_len=None, dtype=np.int32):
    """
    :param batch_words: [batch_size, sent_length]
    :param max_sent_len: if None, max(sent_length)
    :param dtype:
    :return: padding_words: [batch_size, max_sent_length], 0
    """
    if max_sent_len is None:
        max_sent_len = np.max([len(words) for words in response_dec])
    batch_size = len(response_dec)
    padding_dec_words = np.zeros((batch_size, max_sent_len), dtype=dtype)
    padding_target_words = np.zeros((batch_size, max_sent_len), dtype=dtype)
    for i in range(batch_size):
        dec_words= get_dec_seqs(response_dec[i], max_sent_len, word_vocab[start_word])
        target_words = get_target_seqs(response_target[i], max_sent_len, word_vocab[stop_word])
        assert len(dec_words) == len(target_words)
        kept_length = len(dec_words)
        if kept_length > max_sent_len:
            kept_length = max_sent_len
        padding_dec_words[i, :kept_length] = dec_words[:kept_length]
        padding_target_words[i, :kept_length] = target_words[:kept_length]
    return padding_dec_words, padding_target_words


def fill_2d_matrix(batch_f_rf, num_vocab, dtype=np.float32):
    batch_size = len(batch_f_rf)
    padding_rf = np.zeros((batch_size, num_vocab), dtype=dtype)
    for i in range(batch_size):
        for idx, value in batch_f_rf[i].items():
            padding_rf[i][idx] = value
    return padding_rf


def build_word_vocab(sents, threshold=1):
    """
    :param sents:
    :return: word2index
    """
    dictionary = {}
    for sent in sents:
        for word in sent:
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1
    print(len(sents))
    print(len(dictionary))
    dictionary = removeItemsInDict(dictionary, threshold)
    print(len(dictionary))
    words_vocab = {str(key): index + 4 for index, key in enumerate(sorted(dictionary.keys()))}
    # words_vocab = {word: index+2 for index, word in enumerate(words)}
    words_vocab[pad_word] = 0
    words_vocab[unk_word] = 1
    words_vocab[start_word] = 2
    words_vocab[stop_word] = 3
    # words_vocab = removeItemsInDict(words_vocab, threshold)
    return words_vocab


def build_utter_vocab(utters, threshold=1):
    """
    :param sents:
    :return: word2index
    """
    dictionary = {}
    for utter in utters:
        for sent in utter:
            for word in sent:
                if word not in dictionary:
                    dictionary[word] = 0
                dictionary[word] += 1
    print(len(utters))
    print(len(dictionary))
    dictionary = removeItemsInDict(dictionary, threshold)
    print(len(dictionary))
    words_vocab = {str(key): index + 4 for index, key in enumerate(sorted(dictionary.keys()))}
    # words_vocab = {word: index+2 for index, word in enumerate(words)}
    words_vocab[pad_word] = 0
    words_vocab[unk_word] = 1
    words_vocab[start_word] = 2
    words_vocab[stop_word] = 3
    return words_vocab


def build_ner_vocab(ners):
    """
   :param ners:
   :return: ner2index
   """
    ner_set = set()
    for ner in ners:
        ner_set.update(ner)
    ners_vocab = {ner: index for index, ner in enumerate(ner_set)}
    return ners_vocab


def build_pos_vocab(poses):
    """
   :param ners:
   :return: ner2index
   """
    pos_set = set()
    for pos in poses:
        pos_set.update(pos)
    pos_vocab = {pos: index for index, pos in enumerate(pos_set)}
    return pos_vocab


def build_char_vocab(sents):
    """
    :param sents:
    :return: char2index
    """
    chars = set()
    for sent in sents:
        for word in sent:
            word = list(word)
            chars.update(word)
    chars_vocab = {char: index+2 for index, char in enumerate(chars)}
    chars_vocab[pad_word] = 0
    chars_vocab[unk_word] = 1
    return chars_vocab


def merge_vocab(vacab1, vocab2):
    words_vocab = vacab1.copy()
    num = len(vacab1)
    for word, index in vacab1.items():
        if word not in words_vocab:
            words_vocab[word] = num
            num += 1
    for word, index in vocab2.items():
        if word not in words_vocab:
            words_vocab[word] = num
            num += 1
    return words_vocab


def build_character(word_vocab):
    words = []
    for line in word_vocab:
        line = line.strip()
        words.append(line)
    char = "".join(words)
    char_set = set(char)
    char_dict = {}
    i = 1
    for char in char_set:
        char_dict[char] = i
        i += 1
    return char_dict


def sort_hyps(hyps):
    """"Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def outputids2words(id_list, vocab):
    words = []
    for i in id_list:
        w = vocab[i]  # might be [UNK]
        words.append(w)
    return words


def load_fasttext_unk_words(oov_word_list, word2index, word_embedding):
    pass


def load_fasttext(word2index, emb_file, n_dim=100):
    """
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param word2index: dic, word2index['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :return: np.array(n_words, n_dim)
    """
    pass


def load_word_embedding(word2index, emb_file, config, n_dim=300):
    """
    UPDATE_1: fix the
    ===
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param word2index: dic, word2index['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :return: np.array(n_words, n_dim)
    """
    print('Load word embedding: %s' % emb_file)

    # assert word2index[pad_word] == 0
    # assert word2index[unk_word] == 1

    pre_trained = {}
    n_words = len(word2index)

    embeddings = np.random.uniform(-0.25, 0.25, (n_words, n_dim))
    embeddings[0, ] = np.zeros(n_dim)

    with open(emb_file, 'r') as f:
        for idx, line in enumerate(f):
            # 第一行可能是维度和行数
            if idx == 0 and len(line.split()) == 2:
                continue
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])
            # 词
            w = ''.join(sp[0:len(sp) - n_dim])
            # 词向量
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            if w in word2index and w not in pre_trained:
                embeddings[word2index[w]] = emb
                pre_trained[w] = 1

    pre_trained_len = len(pre_trained)

    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))
    # 未登录词
    oov_word_list = [w for w in word2index if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open(config.oov_file, 'wb'))
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


def load_embed_from_vocab(word_vocab, embed_list):
    word_embed_dict = {}
    for key, value in word_vocab.items():
        word_embed_dict[key] = embed_list[value]
    return word_embed_dict


def load_embed_from_text(emb_file, token_dim):
    """
    :return: embed: numpy, vocab2id: dic
    """
    print('==> loading embed from txt')

    embed = []
    vocab2id = {}

    word_id = 0
    embed.append([0.0] * token_dim)

    with open(emb_file, 'r') as fr:

        print('embedding info: ', fr.readline())

        for line in fr:
            t = line.rstrip().split()
            word_id += 1
            vocab2id[t[0]] = word_id

            # python3 map return a generator not a list
            embed.append(list(map(float, t[1:])))

    print('==> finished load input embed from txt')
    return np.array(embed, dtype=np.float32), vocab2id


def get_word_embed_meta(embed_file, meta_file):
    word = load_params(embed_file)
    word_v = sorted(word.items(), key=lambda x: x[1])
    with open(meta_file, 'w', encoding="utf-8") as f1:
        for key, value in word_v:
            f1.write(key + '\n')


def cout_distribution(examples):
    label_count = np.zeros(20)
    for example in examples:
        label_count[int(example[1])] += 1
    sum = np.sum(label_count)
    for i in label_count:
        print(i / sum * 100), "%", " ", i


def hashtagSegment(segmenter, word):
    token2 = []
    token1 = (segmenter.get(word)).split(" ")  # 对hashtag进行分词
    for word_ in token1:
        if len(word_) >= 2:
            token2.append(word_)
    return token2

def get_gold_label(gold_file):
    refs = []
    with open(gold_file, "r") as f:
        for line in f:
            refs.append(line.strip().split(' '))

class Batch(object):
    """
    Tricks:
    1. setattr and getattr
    2. __dict__ and vars
    """
    def __init__(self):

        pass

    def add(self, name, value):
        setattr(self, name, value)

    def get(self, name):
        if name == 'self':
            value = self.__dict__  # or value = vars(self)
        else:
            value = getattr(self, name)
        return value