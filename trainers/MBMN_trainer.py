# -*- coding:utf-8 _*-
from trainers.SCNMA_trainer import SCNMATrainer


class MBMNTrainer(SCNMATrainer):
    def __init__(self, sess, model, data, config, logger):
        super(MBMNTrainer, self).__init__(sess, model, data, config, logger)

    def train_step(self, batch):
        feed_dict = {
            self.model.input_x_utter: batch.utter,
            self.model.input_x_response: batch.res,
            self.model.input_utter_num: batch.utter_num,
            self.model.input_res_len: batch.res_len,
            self.model.input_utter_len: batch.utter_len,
            self.model.weighted_mask: batch.utter_weighted_mask,
            self.model.enc_padding_mask: batch.enc_padding_mask,
            self.model.final_padding_mask: batch.final_padding_mask,
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
            # 'sim_weighted': self.model.sim_weighted,
            # 'final_padding_mask': self.model.final_padding_mask
        }
        return self.sess.run(to_return, feed_dict)

    def test_step(self, batch):
        feed_dict = {
            self.model.input_x_utter: batch.utter,
            self.model.input_x_response: batch.res,
            self.model.input_utter_num: batch.utter_num,
            self.model.input_res_len: batch.res_len,
            self.model.input_utter_len: batch.utter_len,
            self.model.weighted_mask: batch.utter_weighted_mask,
            self.model.enc_padding_mask: batch.enc_padding_mask,
            self.model.final_padding_mask: batch.final_padding_mask,
            self.model.input_y: batch.label,
            self.model.drop_keep_rate: 1.0,
        }
        if self.config.feature_p:
            feed_dict[self.model.input_x_feature_p] = batch.feature_p
        if self.config.feature_u:
            feed_dict[self.model.input_x_feature_u] = batch.feature_u
        to_return = {
            'predict_label': self.model.predict_label,
            'predict_prob': self.model.predict_prob
        }
        return self.sess.run(to_return, feed_dict)