# -*- coding:utf-8 _*-
from base.base_train import BaseTrain
import numpy as np
from utils import evaluation
import tqdm


class SCNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SCNTrainer, self).__init__(sess, model, data, config, logger)

    def train(self):
        best_dev_R10_1 = 0
        best_dev_result = []
        best_test_result = []
        early_stop = 0
        self.model.load(self.sess)
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            dev_R2_1, dev_R10_1, dev_result = self.do_eval(self.dev_data)
            test_R2_1, test_R10_1, test_result = self.do_eval(self.test_data)
            print('----'*10)
            print('dev_R2_1 = {:.5f}, dev_R10_1 = {:.5f}, test_R2_1 = {:.5f}, test_R10_1 = {:.5f}'.format(dev_R2_1, dev_R10_1, test_R2_1, test_R10_1))
            if dev_R10_1 > best_dev_R10_1:
                best_dev_R10_1 = dev_R10_1
                best_dev_result = dev_result
                best_test_result = test_result
                self.model.save(self.sess)
                print(
                    'dev_R10_1 = {:.5f} best!!!!'.format(best_dev_R10_1)
                    # 'dev = {:.5f} best!!!!'.format(best_dev_macro_f1)
                )
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 3:
                break
        with open(self.config.dev_predict_file, 'w') as f:
            for label in best_dev_result:
                f.write(str(float(label)) + "\n")
        with open(self.config.test_predict_file, 'w') as f:
            for label in best_test_result:
                f.write(str(float(label))+"\n")

    def do_eval(self, data):
        batch_size = 100  # for Simple
        preds, golds = [], []
        for batch in data.next_batch(batch_size):
            results = self.test_step(batch)
            preds.append(results['predict_prob'][:, 1])
            golds.append(np.argmax(batch.label, 1))
        preds = np.concatenate(preds, 0)
        golds = np.concatenate(golds, 0)
        R2_1 = evaluation.ComputeR2_1(preds, golds)
        R10_1 = evaluation.ComputeR10_1(preds, golds)
        return R2_1, R10_1, preds

    def do_test(self):
        self.model.load(self.sess)
        batch_size = 100  # for Simple
        preds, golds = [], []
        batch_num = 0
        for batch in self.test_data.next_batch(batch_size):
            batch_num += 1
            results = self.test_step(batch)
            preds.append(results['predict_prob'][:, 1])
            golds.append(np.argmax(batch.label, 1))
        preds = np.concatenate(preds, 0)
        golds = np.concatenate(golds, 0)
        R10_1 = evaluation.ComputeR10_1(preds, golds)
        with open(self.config.gold_label_file, 'w') as f:
            for label in golds:
                f.write(str(int(label))+"\n")
        with open(self.config.test_predict_file, 'w') as f:
            for label in preds:
                f.write(str(float(label))+"\n")
        print(R10_1)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        losses = []
        total_batch = 0
        for batch in self.train_data.next_batch(self.config.batch_size, shuffle=False):
            total_batch += 1
            results = self.train_step(batch)
            step = self.model.global_step_tensor.eval(self.sess)
            loss = results['loss']
            losses.append(results['loss'])
            if total_batch % self.config.display_step == 0:
                print('batch_{} steps_{} cost_val: {:.5f}'.format(total_batch, step, loss))
                print('==>  Epoch {:02d}/{:02d}'.format(self.model.cur_epoch_tensor.eval(self.sess), total_batch))
                # print('sim_weighted:', results['sim_weighted'][0])
                # print('final_padding_mask:', results['final_padding_mask'][0])
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self, batch):
        feed_dict = {
            self.model.input_x_utter: batch.utter,
            self.model.input_x_response: batch.res,
            self.model.input_utter_num: batch.utter_num,
            self.model.input_res_len: batch.res_len,
            self.model.input_utter_len: batch.utter_len,
            self.model.input_y: batch.label,
            self.model.drop_keep_rate: self.config.drop_keep_rate,
            self.model.learning_rate: self.config.learning_rate
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

    def test_step(self, batch):
        feed_dict = {
            self.model.input_x_utter: batch.utter,
            self.model.input_x_response: batch.res,
            self.model.input_utter_num: batch.utter_num,
            self.model.input_res_len: batch.res_len,
            self.model.input_utter_len: batch.utter_len,
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