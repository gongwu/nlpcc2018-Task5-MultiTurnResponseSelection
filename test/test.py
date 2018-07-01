# -*- coding:utf-8 -*-
from data_preprocess import data_preprocess
from utils import data_utils
import json
# from utils import evaluation
import numpy as np
import tensorflow as tf
import pickle
from utils import tf_utils
from utils import utils
# sample_list = data_preprocess.data_to_json('../data/dialogue_data/word/train.txt')[:1000]
# with open('../data/dialogue_data/word/demo.json', "w") as f:
#     json.dump(sample_list, f, indent=2, ensure_ascii=False)
#     print("加载入文件完成...")
# data_preprocess.word_seg('../data/dialogue_data/word/train_trunc.txt', '../data/dialogue_data/word/train_seg_baidu.txt')

# data_preprocess.write_response('../data/dialogue_data/word/train_seg.txt', '../data/dialogue_data/word/response_seg.txt')

# sessions_list = data_preprocess.sample_response('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/response_seg.txt', 9)
# with open('../data/dialogue_data/word/dev_sample.json', "w") as f:
#     json.dump(sessions_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# data_preprocess.word_split('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/train_seg.txt', '../data/dialogue_data/word/dev_seg.txt', 4960000)
# sessions_train_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/train_seg.txt')
# sessions_dev_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/dev_seg.txt')
# with open('../data/dialogue_data/word/train_seg.json', "w") as f:
#     json.dump(sessions_train_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# with open('../data/dialogue_data/word/dev_seg.json', "w") as f:
#     json.dump(sessions_dev_list, f, indent=2, ensure_ascii=False)
#     print("加载入验证文件完成...")


# sample_list = data_preprocess.data_truncation_seg('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/trunc_seg.txt')
# data_preprocess.word_split('../data/dialogue_data/word/final/final_train_sample2.txt', '../data/dialogue_data/word/final/final_train_sample.txt', '../data/dialogue_data/word/final/final_dev_sample.txt', 920000)
# sessions_train_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/final/final_train.txt')
# sessions_dev_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/final/final_dev.txt')
# sessions_test_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/final/final_test.txt')
# with open('../data/dialogue_data/word/final/final_train.json', "w") as f:
#     json.dump(sessions_train_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# with open('../data/dialogue_data/word/final/final_dev.json', "w") as f:
#     json.dump(sessions_dev_list, f, indent=2, ensure_ascii=False)
#     print("加载入验证文件完成...")
# with open('../data/dialogue_data/word/final/final_test.json', "w") as f:
#     json.dump(sessions_test_list, f, indent=2, ensure_ascii=False)
#     print("加载入验证文件完成...")
# data_preprocess.get_ngram_idf('../data/dialogue_data/word/train_seg.json', '../data/dialogue_data/word/dev_seg.json', '../experiments/generator/Seq2SeqAttention_double_trunc/dic/idf.txt', threshold = 1)
# idf_dic = data_utils.load_params('../experiments/generator/Seq2SeqAttention_double_trunc/dic/idf.txt')
# data_preprocess.get_entropy(idf_dic, '../data/dialogue_data/word/response_seg.txt', '../experiments/generator/Seq2SeqAttention_double_trunc/dic/entropy.txt')
# entropy_dic = data_utils.load_params('../experiments/generator/Seq2SeqAttention_double_trunc/dic/entropy.txt')
# sample_list = data_preprocess.data_truncation_seg_with_entropy('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/trunc_seg_with_entropy.txt', entropy_dic)
# data_preprocess.word_seg('../data/dialogue_data/word/train.txt', '../data/dialogue_data/word/train_seg_hg.txt')

# from pyltp import Segmentor
# segmentor = Segmentor()  # 初始化实例
# segmentor.load(cws_model_path)  # 加载模型
# words = segmentor.segment('一个超长音在试了一遍才知道吼也是要有技术的')  # 分词
# print (' '.join(words))
# segmentor.release()  # 释放模型

# sessions_test_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/final/final_test.txt')
# with open('../data/dialogue_data/word/final/test_response.txt', "w") as f:
#     for session in sessions_test_list:
#         decoded_output = ' '.join(session['response'])
#         f.write(decoded_output)
#         f.write('\n')


# data_preprocess.get_ngram_idf('../data/dialogue_data/word/final/final_train.json', '../data/dialogue_data/word/final/final_dev.json', '../experiments/generator/Seq2SeqAttention_double_final/dic/idf.txt', threshold = 1)
# idf_dic = data_utils.load_params('../experiments/generator/Seq2SeqAttention_double_final/dic/idf.txt')
# for key, value in idf_dic.items():
#     print(key)
# sample_list = data_preprocess.data_truncation_seg('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/final/final_total_selection.txt')
# sessions_train_list = data_preprocess.sample_response('../data/dialogue_data/word/final/final_train_sample.txt', '../data/dialogue_data/word/response_seg.txt', 1)
# with open('../data/dialogue_data/word/final/final_train_sample.json', "w") as f:
#     json.dump(sessions_train_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# sessions_devlist = data_preprocess.sample_response('../data/dialogue_data/word/final/final_dev_sample.txt', '../data/dialogue_data/word/response_seg.txt', 9)
# with open('../data/dialogue_data/word/final/final_dev_sample.json', "w") as f:
#     json.dump(sessions_devlist, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# sessions_testlist = data_preprocess.sample_response('../data/dialogue_data/word/final/final_test_sample.txt', '../data/dialogue_data/word/response_seg.txt', 9)
# with open('../data/dialogue_data/word/final/final_test_sample.json', "w") as f:
#     json.dump(sessions_testlist, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")


# sample_list = data_preprocess.data_truncation_random('../data/dialogue_data/word/total_seg.txt', '../data/dialogue_data/word/final/final_total_selection.txt')

# with tf.Session() as sess:
#     a = tf.constant([[1, 2, 0],[1, 3, 4]], dtype=tf.int32)
#     b = tf.constant([2, 3], dtype=tf.int32)
#     c = tf_utils.Mask([[[1, 2, 3],[1, 3, 4], [1, 1, 1]],[[1, 3, 5], [2, 3, 5], [3, 4, 5]]], [2, 3], 3)
#     d = tf.reduce_sum(c, axis=1)
#     print(sess.run(c))
#     print(sess.run(d))

# data_preprocess.word_seg('../data/dialogue_data/word/test.txt', '../data/dialogue_data/word/test_seg.txt')

# sessions_test_list = data_preprocess.data_to_json('../data/dialogue_data/word/test_seg.txt')
# with open('../data/dialogue_data/word/test.json', "w") as f:
#     json.dump(sessions_test_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# data_preprocess.get_ngram_idf('../data/dialogue_data/word/final/final_train_sample.json', '../data/dialogue_data/word/final/final_test_sample.json', '../data/dialogue_data/word/final/idf.txt', threshold = 1)

# examples = data_utils.read_dialogue_selection_data('../data/dialogue_data/word/final/final_test_sample.json')
# with open('../data/dialogue_data/word/final/gold_label_file.txt', "w") as f:
#     for example in examples:
#         f.write(str(example[2]) + '\n')
# data_preprocess.show_result_generator('../data/dialogue_data/word/test_seg.txt', '../experiments/generator/Seq2Seq_final_2/output/test_predict_file_beam.txt', '../experiments/generator/Seq2Seq_final_2/output/test_predict_file_beam_all.txt')

# def build_vocab(train_file, dev_file, test_file):
#     if test_file is None:
#         print('test_file is None')
#         file_list = [train_file, dev_file]
#     else:
#         file_list = [train_file, dev_file, test_file]
#     examples = data_utils.read_dialogue_generator_data(file_list)
#     utters = []
#     responses = []
#     for example in examples:
#         utter = example[0]
#         response = example[1]
#         utters.append(utter)
#         responses.append(response)
#     # 这里必须将vocab_res放在vocab_utter的后面，不然在decoder阶段会出现index不对应
#     # 统计平均长度与最大长度
#     max_sent_len = 0
#     avg_sent_len = 0
#     for sent in responses:
#         if len(sent) > max_sent_len:
#             max_sent_len = len(sent)
#         avg_sent_len += len(sent)
#     avg_sent_len /= len(responses)
#     print('task: max_sent_len: {}'.format(max_sent_len))
#     print('task: avg_sent_len: {}'.format(avg_sent_len))
#     max_utters_num = 0
#     avg_utters_num = 0
#     for utter in utters:
#         if len(utter) > max_utters_num:
#             max_utters_num = len(utter)
#         avg_utters_num += len(utter)
#     avg_utters_num /= len(utters)
#     print('task: max_utters_num: {}'.format(max_utters_num))
#     print('task: avg_utters_num: {}'.format(avg_utters_num))
#
# build_vocab('../data/dialogue_data/word/train_seg.json', '../data/dialogue_data/word/dev_seg.json', None)
#
# data_preprocess.data_to_test_selection('../data/dialogue_data/word/seq_context.txt', '../data/dialogue_data/word/seq_replies.txt', '../data/dialogue_data/word/test_selection.txt')
# data_preprocess.word_seg('../experiments/generator/result/sub1-reply.txt', '../experiments/generator/result/sub1-reply-seg.txt')
# sessions_test_list = data_preprocess.data_to_json_seg('../data/dialogue_data/word/test_selection_seg.txt')
# with open('../data/dialogue_data/word/test_selection.json', "w") as f:
#     json.dump(sessions_test_list, f, indent=2, ensure_ascii=False)
#     print("加载入训练文件完成...")
# data_preprocess.show_result_generator('../data/dialogue_data/word/test.txt', '../experiments/generator/S2SAMSE_final/output/sub1-reply-seg.txt', '../experiments/generator/S2SAMSE_final/output/sub1-reply-seg-all.txt')

# data_preprocess.word_seg('../experiments/generator/S2SAMSE_final/output/ECNU.txt', '../experiments/generator/S2SAMSE_final/output/ECNU_seg.txt')
# from utils import nltk_bleu_score
# hyps = []
# refs = []
# # reference = [['今天', '天气', '真好', '耶']]
# # candidate = ['今天', '天气', '真好', '啊']
# # hyps.append(candidate)
# # refs.append(reference)
# # reference = [['我', '是', '猫', '啊']]
# # candidate = ['我', '是', '猫', '啊']
# # hyps.append(candidate)
# # refs.append(reference)
# with open('../experiments/generator/result/pred.txt', "r") as f:
#     for line in f:
#         hyps.append(line.strip().split(' '))
#
# with open('../experiments/generator/result/sub1-reply-seg.txt', "r") as f:
#     for line in f:
#         refs.append([line.strip().split(' ')])
# # score1 = evaluation.get_bleu_1(refs, hyps)
# # score2 = evaluation.get_bleu_2(refs, hyps)
# # score3 = evaluation.get_bleu_3(refs, hyps)
# # score4 = evaluation.get_bleu_4(refs, hyps)
# # score = evaluation.get_bleu_avg(refs, hyps)
# score_all = nltk_bleu_score.corpus_bleu(refs, hyps)
# print(score_all)
# data_preprocess.get_ngram_idf('../data/dialogue_data/word/train_seg.json', '../data/dialogue_data/word/dev_seg.json', '../experiments/generator/Seq2SeqAttention_double_final/dic/idf_all.txt', threshold = 2)
data_preprocess.get_selection_result('../data/dialogue_data/word/seq_replies.txt', '../experiments/dialogue/MBMN_final_all_SCN_st/output/test_predict_file.txt', '../experiments/dialogue/MBMN_final_all_SCN_st/output/test_index.txt')
# data_preprocess.show_selection_result('../data/dialogue_data/word/seq_context.txt', '../data/dialogue_data/word/seq_replies.txt', '../experiments/dev_index_ensemble_new.txt', '../experiments/dev_predict_file_ensemble_new_all.txt')
# data_preprocess.repalce_generate('../experiments/generator/S2SAM_final/output/test_predict_file_greedy_all.txt', '../experiments/generator/S2SAM_final/output/test_predict_file_all.txt', '../experiments/generator/S2SAM_final/output/test_predict_file_all_merge.txt', '../experiments/generator/S2SAM_final/output/a.txt', '../experiments/generator/S2SAM_final/output/b.txt')
# data_preprocess.ensemble('../experiments/model_all/dev_predict_file.txt', '../experiments/model_MEMORY/dev_predict_file.txt',
#                          '../experiments/model_SCN/dev_predict_file.txt','../experiments/model_feature/dev_predict_file.txt',
#                          '../experiments/dev_predict_file_ensemble_new.txt')
score = data_preprocess.p1('../experiments/dialogue/result/sub2-index.txt', '../experiments/dialogue/MBMN_final_all_SCN_st/output/test_index.txt')
print(score)
# list_index = data_preprocess.case_study('../experiments/dialogue/result/test_index_memory.txt', '../experiments/dialogue/result/test_index_SCN.txt', '../experiments/dialogue/result/sub2-index.txt')
# data_preprocess.case_study_show('../experiments/dialogue/result/test_predict_file_memory.txt', '../experiments/dialogue/result/test_predict_file_SCN.txt', '../experiments/dialogue/result/sub2-index.txt', '../experiments/dialogue/result/case_study_1.txt', list_index)
# W = tf.Variable(np.ones((5,))[None, :, None], name='static_weight', dtype=tf.float32)
# with tf.Session() as sess:
#     M_s = tf.constant([1, 1])
#     M_s = tf.expand_dims(tf.expand_dims(M_s, 0), 0)
#     print(sess.run(M_s))
