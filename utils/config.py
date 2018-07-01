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
    ROOT = '../experiments'
    DATA_DIR = '../data/twitter_data/English'
    VOCABULARY_DIR = '../vocabulary'
    DIC_DIR = DATA_DIR + '../dic'
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join(ROOT, config.exp_name, "summary")
    config.checkpoint_dir = os.path.join(ROOT, config.exp_name, "checkpoint/")
    config.train_file = os.path.join(DATA_DIR, "processed", "2of3.json")
    config.dev_file = os.path.join(DATA_DIR, "processed", "1of3.json")
    config.test_file = None
    config.dev_predict_file = os.path.join(ROOT, config.exp_name, "output/dev_predict_file.txt")
    config.word_embed_file = os.path.join(DATA_DIR, "embed", "SWM.vocab.vector")
    config.w2i_file = os.path.join(DIC_DIR, "w2i.p")
    config.c2i_file = os.path.join(DIC_DIR, "c2i.p")
    config.n2i_file = os.path.join(DIC_DIR, "n2i.p")
    config.p2i_file = os.path.join(DIC_DIR, "p2i.p")
    config.oov_file = os.path.join(DIC_DIR, "oov.p")
    config.we_file = os.path.join(DIC_DIR, "we.p")
    config.VOCAB_NORMAL_WORDS_PATH = os.path.join(VOCABULARY_DIR, "normal_word.pkl")
    config.max_sent_len = 34
    config.max_word_len = 38
    config.num_class = 20
    config.word_dim = 300
    config.char_dim = 50
    config.ner_dim = 50
    config.pos_dim = 50
    config.category2id = {"_red_heart_": 0, "_smiling_face_with_hearteyes_": 1, "_face_with_tears_of_joy_": 2, "_two_hearts_": 3,
               "_fire_": 4, "_smiling_face_with_smiling_eyes_": 5, "_smiling_face_with_sunglasses_": 6, "_sparkles_": 7,
               "_blue_heart_": 8, "_face_blowing_a_kiss_": 9, "_camera_": 10, "_United_States_": 11, "_sun_": 12,
               "_purple_heart_": 13, "_winking_face_": 14, "_hundred_points_": 15, "_beaming_face_with_smiling_eyes_": 16,
               "_Christmas_tree_": 17, "_camera_with_flash_": 18, "_winking_face_with_tongue_": 19}
    config.id2category = {index: label for label, index in config.category2id.items()}
    return config
