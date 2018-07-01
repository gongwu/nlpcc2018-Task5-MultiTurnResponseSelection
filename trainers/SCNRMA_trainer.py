# -*- coding:utf-8 _*-
from trainers.SCNMA_trainer import SCNMATrainer


class SCNRMATrainer(SCNMATrainer):
    def __init__(self, sess, model, data, config, logger):
        super(SCNMATrainer, self).__init__(sess, model, data, config, logger)