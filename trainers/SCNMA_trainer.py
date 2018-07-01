# -*- coding:utf-8 _*-
from trainers.SCN_trainer import SCNTrainer


class SCNMATrainer(SCNTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(SCNMATrainer, self).__init__(sess, model, data, config, logger)

    def train_step(self, batch):
        feed_dict = {
            self.model.input_x_utter: batch.utter,
            self.model.input_x_response: batch.res,
            self.model.input_utter_num: batch.utter_num,
            self.model.input_res_len: batch.res_len,
            self.model.input_utter_len: batch.utter_len,
            self.model.input_y: batch.label,
            self.model.drop_keep_rate: self.config.drop_keep_rate,
            self.model.learning_rate: self.config.learning_rate,
        }
        if self.config.feature_p:
            feed_dict[self.model.input_x_feature_p] = batch.feature_p
        if self.config.feature_u:
            feed_dict[self.model.input_x_feature_u] = batch.feature_u
        to_return = {
            'train_step': self.model.train_step,
            'loss': self.model.loss,
        }
        return self.sess.run(to_return, feed_dict)
