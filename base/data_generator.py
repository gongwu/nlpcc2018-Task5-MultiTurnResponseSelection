import numpy as np
from utils import data_utils
from utils import utils


class DataGenerator(object):
    def __init__(self, config):
        self.config = config

    def build_data(self):
        """
            return the formatted matrix, which is used as the input to deep learning models
            Args: file_list:
              word_vocab:
        """
        raise NotImplementedError

    def next_batch(self):
        raise NotImplementedError

    def build_vocab(self):
        """
            build sents is for build vocab
            during multi-lingual task, there are two kinds of sents
            :return: sents
        """
        raise NotImplementedError

    def init_embedding(self):
        raise NotImplementedError

