# coding: utf-8
import pandas as pd
from configs import config_twitter
from utils.confusion_matrix import Alphabet, ConfusionMatrix
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

DICT_LABEL_TO_INDEX = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4, 'world': 5, 'finance': 6, 'game': 7,
                            'travel': 8, 'military': 9, 'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
                             'story': 15, 'regimen': 16, 'essay': 17}

DICT_INDEX_TO_LABEL = {index: label for label, index in DICT_LABEL_TO_INDEX.items()}


def confusion_matrix(gold, pred):
    labels = sorted(set(list(gold) + list(pred)), reverse=False)
    line_0 = ["%5d" % t for t in labels]
    print("Gold\\Pred\t| " + " \t".join(line_0))

    for cur in labels:
        count = dict(((l, 0) for l in labels))
        for i in range(len(gold)):
            if gold[i] != cur: continue
            count[pred[i]] += 1
        count = ["%5d" % (count[l]) for l in labels]
        print("\t%5d\t| " % cur + " \t".join(count))


def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        gold_list = [int(line.strip().split('\t')[0]) for line in gold_file]
        predicted_list = [int(line.strip().split("\t")[0]) for line in predict_file]
        predict_labels = [config_twitter.id2category[int(predict)] for predict in predicted_list]
        gold_labels = [config_twitter.id2category[int(gold)] for gold in gold_list]
        binary_alphabet = Alphabet()
        for i in range(18):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predict_labels, gold_labels)

        confusion_matrix(gold_list, predicted_list)
        cm.print_summary()
        macro_p, macro_r, macro_f1 = cm.get_average_prf()
        overall_accuracy = cm.get_accuracy()
        return overall_accuracy, macro_p, macro_r, macro_f1


def Evaluation_lst(gold_label, predict_label, print_all=False):
    binary_alphabet = Alphabet()
    for i in range(18):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)

    if print_all:
        cm.print_out()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy


def Evaluation_all(gold_label, predict_label):
    binary_alphabet = Alphabet()
    for i in range(18):
        binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

    cm = ConfusionMatrix(binary_alphabet)
    cm.add_list(predict_label, gold_label)
    macro_p, macro_r, macro_f1 = cm.get_average_prf()
    overall_accuracy = cm.get_accuracy()
    return overall_accuracy, macro_p, macro_r, macro_f1


def case_study(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        # gold_list = []
        # for line in gold_file:
        #     gold_list.append([int(line.strip().split('\t')[0]), line.strip().split('\t')[1]])
        gold_list = [int(line.strip().split('\t')[0]) for line in gold_file]
        predicted_list = [int(line.strip().split("\t")[0]) for line in predict_file]
        error_list = []
        for i in range(len(gold_list)):
            if gold_list[i][0] != predicted_list[i]:
                error_list.append(i)
        case_list = []
        for i in error_list:
            case_list.append({'gold': DICT_INDEX_TO_LABEL[gold_list[i][0]]+" "+str(gold_list[i][0]),  'content': gold_list[i][1], 'predicted': DICT_INDEX_TO_LABEL[predicted_list[i]]+" "+str(predicted_list[i])})
        case_data = pd.DataFrame(case_list)
        return case_data


def ComputeR10_1(scores, labels, count=10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total += 1
            sublist = scores[i:i + count]
            if max(sublist) == scores[i]:
                correct += 1
    return float(correct) / total


def ComputeR2_1(scores, labels, count=2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total += 1
            sublist = scores[i:i + count]
            if max(sublist) == scores[i]:
                correct += 1
    return float(correct) / total


def ComputeP_1(scores, labels, count = 10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            sublist = scores[i:i + count]
            total += 1
            if max(sublist) == scores[i]:
                correct += 1
    return float(correct) / total


def get_bleu_1(refs, hyps):
    scores = []
    for i in range(len(hyps)):
        try:
            scores.append(sentence_bleu(refs[i], hyps[i], smoothing_function=SmoothingFunction().method7,
                                        weights=[1, 0, 0, 0]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)

def get_bleu_2(refs, hyps):
    scores = []
    for i in range(len(hyps)):
        try:
            scores.append(sentence_bleu(refs[i], hyps[i], smoothing_function=SmoothingFunction().method7,
                                        weights=[0, 1, 0, 0]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)

def get_bleu_3(refs, hyps):
    scores = []
    for i in range(len(hyps)):
        try:
            scores.append(sentence_bleu(refs[i], hyps[i], smoothing_function=SmoothingFunction().method7,
                                        weights=[0, 0, 1, 0]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)

def get_bleu_4(refs, hyps):
    scores = []
    for i in range(len(hyps)):
        try:
            scores.append(sentence_bleu(refs[i], hyps[i], smoothing_function=SmoothingFunction().method7,
                                        weights=[0, 0, 0, 1]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)

def get_bleu_avg(refs, hyps):
    scores = []
    for i in range(len(hyps)):
        try:
            scores.append(sentence_bleu(refs[i], hyps[i], smoothing_function=SmoothingFunction().method7,
                                        weights=[0.25, 0.25, 0.25, 0.25]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)



# if __name__ == '__main__':
#     dev_overall_accuracy, dev_macro_p, dev_macro_r, dev_macro_f1 = Evaluation(config.dev_predict_file,
#                                                               config.dev_gold_final_file)
#     test_overall_accuracy, test_macro_p, test_macro_r, test_macro_f1 = Evaluation(config.test_predict_file,
#                                                               config.test_gold_file)
#     print('dev: overall_accuracy = {:.5f}, macro_p = {:.5f}, macro_r = {:.5f}, macro_f1 = {:.5f}\n'
#           'test: overall_accuracy = {:.5f}, macro_p = {:.5f}, macro_r = {:.5f}, macro_f1 = {:.5f}\n'
#           .format(dev_overall_accuracy, dev_macro_p, dev_macro_r, dev_macro_f1, test_overall_accuracy, test_macro_p, test_macro_r, test_macro_f1))
    # case_list = case_study(config.test_predict_file, config.test_gold_file)
    # pd.set_option('display.width', 1000)
    # print(case_list.head(10000))

