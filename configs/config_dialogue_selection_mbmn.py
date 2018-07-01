# -*- coding:utf-8 _*-
import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(jsonfile):
    # Root dir
    ROOT = 'experiments/dialogue'
    DATA_DIR = 'data/dialogue_data'
    VOCABULARY_DIR = 'vocabulary'
    # dir path
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join(ROOT, config.exp_name, "summary")
    config.checkpoint_dir = os.path.join(ROOT, config.exp_name, "checkpoint/")
    config.dic_dir = os.path.join(ROOT, config.exp_name, "dic")
    config.result_dir = os.path.join(ROOT, config.exp_name, "output")
    # file path
    config.train_file = os.path.join(DATA_DIR, "word/sample", "train_sample.json")
    config.dev_file = os.path.join(DATA_DIR, "word/sample", "dev_sample.json")
    config.test_file = os.path.join(DATA_DIR, "word", "test_selection.json")
    config.dev_predict_file = os.path.join(config.result_dir, "dev_predict_file.txt")
    config.test_predict_file = os.path.join(config.result_dir, "test_predict_file.txt")
    config.gold_label_file = os.path.join(config.result_dir, "gold_label_file.txt")
    config.word_embed_file = os.path.join(DATA_DIR, "embed", "word2vec_dialogue_200_2.txt")
    config.idf_file = os.path.join(DATA_DIR, "dic", "idf_all.txt")
    config.w2i_file = os.path.join(config.dic_dir, "w2i.p")
    config.r2i_file = os.path.join(config.dic_dir, "r2i.p")
    config.u2i_file = os.path.join(config.dic_dir, "u2i.p")
    config.oov_file = os.path.join(config.dic_dir, "oov.p")
    config.we_file = os.path.join(config.dic_dir, "we.p")
    config.VOCAB_NORMAL_WORDS_PATH = os.path.join(VOCABULARY_DIR, "normal_word.pkl")
    # param
    config.max_sent_len = 25
    config.max_utter_len = 10
    config.num_class = 2
    config.word_dim = 200
    config.eps = 1e-6
    # label
    config.category2id = {'negative': 0, 'positive': 1}
    config.id2category = {index: label for label, index in config.category2id.items()}
    return config
