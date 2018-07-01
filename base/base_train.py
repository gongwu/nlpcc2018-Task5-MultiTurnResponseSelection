import tensorflow as tf
from tqdm import tqdm
from utils import evaluation
import numpy as np
import copy


class BaseTrain(object):
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        if not self.config.forword_only:
            data.build_data(self.config.train_file)
            self.train_data = copy.deepcopy(data)
            data.build_data(self.config.dev_file)
            self.dev_data = copy.deepcopy(data)
        data.build_data(self.config.test_file)
        self.test_data = copy.deepcopy(data)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        best_dev_macro_f1 = 0
        best_dev_result = []
        best_test_result = []
        early_stop = 0
        self.model.load(self.sess)
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            dev_macro_f1, dev_result = self.do_eval(self.dev_data)
            test_macro_f1, test_result = self.do_eval(self.test_data)
            print('dev = {:.5f}, test = {:.5f}'.format(dev_macro_f1, test_macro_f1))
            if dev_macro_f1 > best_dev_macro_f1:
                best_dev_macro_f1 = dev_macro_f1
                best_dev_result = dev_result
                best_test_result = test_result
                self.model.save(self.sess)
                print(
                    'dev = {:.5f} best!!!!, test = {:.5f}'.format(best_dev_macro_f1, test_macro_f1)
                    # 'dev = {:.5f} best!!!!'.format(best_dev_macro_f1)
                )
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 5:
                break
        with open(self.config.dev_predict_file, 'w') as f:
            for label in best_dev_result:
                f.write(str(int(label)) + "\n")
        with open(self.config.test_predict_file, 'w') as f:
            for label in best_test_result:
                f.write(str(int(label))+"\n")

    def do_eval(self, data):
        batch_size = 100  # for Simple
        preds, golds = [], []
        for batch in data.next_batch(batch_size):
            results = self.test_step(batch)
            preds.append(results['predict_label'])
            golds.append(np.argmax(batch.label, 1))
        preds = np.concatenate(preds, 0)
        golds = np.concatenate(golds, 0)
        predict_labels = [self.config.id2category[predict] for predict in preds]
        gold_labels = [self.config.id2category[gold] for gold in golds]
        overall_accuracy, macro_p, macro_r, macro_f1 = evaluation.Evaluation_all(predict_labels, gold_labels)
        return macro_f1, preds

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        -add any summaries you want using the summary
        """
        losses = []
        total_batch = 0
        for batch in self.train_data.next_batch(self.config.batch_size, shuffle=True):
            total_batch += 1
            self.sess.run(self.model.increment_global_step_tensor)
            step = self.model.global_step_tensor.eval(self.sess)
            results = self.train_step(batch)
            loss = results['loss']
            losses.append(results['loss'])
            if total_batch % self.config.display_step == 0:
                print('batch_{} steps_{} cost_val: {:.5f}'.format(total_batch, step, loss))
                print('==>  Epoch {:02d}/{:02d}'.format(self.model.cur_epoch_tensor.eval(self.sess), total_batch))
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def test_step(self):
        """
        implement the logic of the test step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
