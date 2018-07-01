# -*- coding:utf-8 _*-
import codecs
# import jieba
import random
import json
from collections import Counter
import math
from utils import data_utils
import os
from utils import evaluation
# from pyltp import Segmentor
# from aip import AipNlp
# """ 你的 APPID AK SK """
# APP_ID = '11048457'
# API_KEY = 'XVM9RGdLFbs8g89UmA4UkU6q'
# SECRET_KEY = 'Q6IY3iev91Ysms1otCQ5LUxDWlEmvuaP'
# client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
LTP_DATA_DIR = '../tools/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
# segmentor = Segmentor()  # 初始化实例


def data_truncation(read_file_name, write_file_name):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        n_total = 0
        total_list = []
        utterance = []
        n_utterance = 0
        flag = True
        for line in rf:
            if len(line) > 50:
                flag = False
            if n_utterance > 10:
                flag = False
            if line != "\n":
                utterance.append(line.strip())
            else:
                if not flag:
                    utterance = []
                    n_utterance = 0
                    flag = True
                    continue
                for utter in utterance:
                    wf.write(utter+"\n")
                wf.write('\n')
                utterance = []
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


def data_to_json_truncation(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            n_total = 0
            total_list = []
            sessions = {}
            utterance = []
            n_utterance = 0
            flag = True
            for line in f:
                if len(line) > 10:
                    flag = False
                if n_utterance > 5:
                    flag = False
                if line != "\n":
                    utterance.append(line.strip())
                else:
                    if not flag:
                        utterance = []
                        sessions = {}
                        n_utterance = 0
                        flag = True
                        continue
                    sessions["utterance"] = utterance[:n_utterance-1]
                    sessions["response"] = utterance[n_utterance-1]
                    total_list.append(sessions)
                    utterance = []
                    sessions = {}
                    n_utterance = 0
                    n_total += 1
                    continue
                n_utterance += 1
    print(n_total)
    return total_list


def data_to_json(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            n_total = 0
            total_list = []
            sessions = {}
            utterance = []
            n_utterance = 0
            for line in f:
                if line != "\n":
                    utterance.append(line.strip().split(" "))
                else:
                    sessions["utterance"] = utterance
                    sessions["response"] = ""
                    sessions["label"] = -1
                    total_list.append(sessions)
                    utterance = []
                    sessions = {}
                    n_utterance = 0
                    n_total += 1
                    continue
                n_utterance += 1
    print(n_total)
    return total_list


def word_seg(read_file_name, write_file_name):
    # segmentor.load(cws_model_path)  # 加载模型
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        for line in rf:
            if line != "\n":
                line_list = jieba.cut(line)
                wf.write(" ".join(line_list))
                # words = segmentor.segment(line)  # 分词
                # wf.write(' '.join(words) + '\n')
            else:
                wf.write(line)
        # segmentor.release()


def word_split(read_file_name, write_file1, write_file2, count):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file1, 'w', encoding='utf8') as wf1, codecs.open(write_file2, 'w', encoding='utf8') as wf2:
        n = 0
        for line in rf:
            if n < count:
                if line != "\n":
                    wf1.write(line)
                else:
                    n += 1
                    wf1.write(line)
            else:
                if line != "\n":
                    wf2.write(line)
                else:
                    n += 1
                    wf2.write(line)


def data_to_json_seg(file_list):
    if type(file_list) != list:
        file_list = [file_list]
    for file in file_list:
        with codecs.open(file, 'r', encoding='utf8') as f:
            n_total = 0
            total_list = []
            sessions = {}
            utterance = []
            n_utterance = 0
            for line in f:
                if line != "\n":
                    utterance.append(line.strip().split(" "))
                else:
                    sessions["utterance"] = utterance[:n_utterance-1]
                    sessions["response"] = utterance[n_utterance-1]
                    sessions["label"] = -1
                    total_list.append(sessions)
                    utterance = []
                    sessions = {}
                    n_utterance = 0
                    n_total += 1
                    continue
                n_utterance += 1
    print(n_total)
    return total_list


def data_truncation_seg(read_file_name, write_file_name):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        n_total = 0
        total_list = []
        utterance = []
        n_utterance = 0
        flag = True
        for line in rf:
            if len(line.strip().split(" ")) > 20:
                flag = False
            # if n_utterance > 5:
            #     flag = False
            if line != "\n":
                utterance.append(line.strip())
            else:
                if len(utterance[-1].split(" ")) < 4:
                    flag = False
                if n_utterance <2 or n_utterance > 10:
                    flag = False
                if not flag:
                    utterance = []
                    n_utterance = 0
                    flag = True
                    continue
                for utter in utterance:
                    wf.write(utter+"\n")
                wf.write('\n')
                utterance = []
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


def data_truncation_random(read_file_name, write_file_name):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        n_total = 0
        total_list = []
        utterance = []
        n_utterance = 0
        flag = True
        for line in rf:
            if line != "\n":
                utterance.append(line.strip())
            else:
                if n_total % 5 !=  0:
                    flag = False
                if not flag:
                    utterance = []
                    n_utterance = 0
                    flag = True
                    n_total += 1
                    continue
                for utter in utterance:
                    wf.write(utter+"\n")
                wf.write('\n')
                utterance = []
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


def data_truncation_seg_with_entropy(read_file_name, write_file_name, entropy_dic):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        n_total = 0
        total_list = []
        utterance = []
        n_utterance = 0
        flag = True
        for line in rf:
            if len(line.strip().split(" ")) > 20:
                flag = False
            if line != "\n":
                utterance.append(line.strip())
            else:
                if len(utterance[-1].split(" ")) < 5:
                    flag = False
                if len(utterance[-1].split(" ")) > 15:
                    flag = False
                if entropy_dic[utterance[-1]] < 5.0:
                    flag = False
                if n_utterance > 5:
                    flag = False
                if not flag:
                    utterance = []
                    n_utterance = 0
                    flag = True
                    continue
                for utter in utterance:
                    wf.write(utter+"\n")
                wf.write('\n')
                utterance = []
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


def write_response(read_file_name, write_file_name):
    with codecs.open(read_file_name, 'r', encoding='utf8') as rf, codecs.open(write_file_name, 'w', encoding='utf8') as wf:
        n_total = 0
        total_list = []
        utterance = []
        n_utterance = 0
        for line in rf:
            if line != "\n":
                utterance.append(line.strip())
            else:
                wf.write(utterance[n_utterance-1]+'\n')
                utterance = []
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


def get_response_list(file_name):
    response_list = []
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            response_list.append(line)
    return response_list


def sample_response(data_file_name, response_filename, sample_num):
    response_list = get_response_list(response_filename)
    with codecs.open(data_file_name, 'r', encoding='utf8') as f:
        n_total = 0
        total_list = []
        sessions = {}
        sessions_sample = {}
        utterance = []
        n_utterance = 0
        for line in f:
            if line != "\n":
                utterance.append(line.strip().split(" "))
            else:
                # if n_total < 4960000:
                #     n_total += 1
                #     utterance = []
                #     sessions = {}
                #     n_utterance = 0
                #     continue
                sessions["utterance"] = utterance[:n_utterance-1]
                sessions["response"] = utterance[n_utterance-1]
                sessions["label"] = 1
                total_list.append(sessions)
                for i in range(sample_num):
                    random_response = random.randint(0, len(response_list)-1)
                    sessions_sample["utterance"] = utterance[:n_utterance-1]
                    sessions_sample["response"] = response_list[random_response].strip().split(" ")
                    sessions_sample["label"] = 0
                    total_list.append(sessions_sample)
                    sessions_sample = {}
                utterance = []
                sessions = {}
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
    print(n_total)
    return total_list


'''计算idf'''
def get_ngram_idf(train_data_path, devtest_data_path, to_file, threshold=1):
    print("==" * 40)
    print("==" * 40)

    train_microblogs = json.load(open(train_data_path, "r"), encoding="utf-8")
    devtest_microblogs = json.load(open(devtest_data_path, "r"), encoding="utf-8")
    all_microblogs = train_microblogs + devtest_microblogs
    total = len(all_microblogs)
    # 统计词在文档中出现的频次
    IDF_Counter = Counter()

    # vocab = set([])
    vocab = {}
    for microblog in all_microblogs:
        for word in set(microblog["response"]):
            # if float(microblog["sentiment score"]) < 0:
            IDF_Counter[word] += 1
            # vocab.add(word)
            data_utils.set_dict_key_value(vocab, word)
    # 删除频率小于threshold的键
    data_utils.removeItemsInDict(vocab, threshold)

    # rf
    dict_idf = {}
    for word in vocab:
        dict_idf[word] = math.log(total/ float(IDF_Counter[word]+1))
    # dict_idf = sorted(dict_idf.items(), key=lambda e: e[1], reverse=False)
    data_utils.save_params(dict_idf, to_file)


'''计算信息熵'''
def get_entropy(idf_dic, from_file, to_file):
    with codecs.open(from_file, 'r', encoding='utf8') as f1:
        dict_entropy = {}
        for line in f1:
            line_list = line.strip().split(" ")
            entropy = 0.0
            for word in line_list:
                entropy += idf_dic[word]
            dict_entropy[line.strip()] = entropy/len(line_list)
        data_utils.save_params(dict_entropy, to_file)
        # dict_entropy = sorted(dict_entropy.items(), key=lambda x: x[1], reverse=False)
        # to_file = open(to_file, "w")
        # to_file.write("\n".join(["%s\t%.4f" % (key, value) for key, value in dict_entropy]))
        # to_file.close()


def dict2list(dic:dict):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def show_result_generator(source_file, result_file, to_file):
    with open(source_file, "r") as f1, open(result_file, "r") as f2, open(to_file, "w") as f3:
        n_total = 0
        total_list = []
        sessions = {}
        utterance = []
        n_utterance = 0
        for line in f1:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions["utterance"] = utterance
                total_list.append(sessions)
                utterance = []
                sessions = {}
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
        response = []
        for line in f2:
            response.append(line.strip())
        assert len(total_list) == len(response)
        for i in range(len(total_list)):
            for utter in total_list[i]["utterance"]:
                f3.write(utter+"\n")
            f3.write(response[i]+"\n")
            f3.write('\n')

def data_to_test_selection(context_file, reply_file, to_file):
    with codecs.open(context_file, 'r', encoding='utf8') as f1, codecs.open(reply_file, 'r', encoding='utf8') as f2, codecs.open(to_file, 'w', encoding='utf8') as f3:
        n_total_utter = 0
        sessions = []
        utterance = []
        for line in f1:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions.append(utterance)
                utterance = []
                n_total_utter += 1
        print(n_total_utter)
        replies = []
        reply = []
        n_reply = 0
        for line in f2:
            if line != "\n":
                reply.append(line.strip())
            else:
                replies.append(reply)
                if len(reply) != 10:
                    print(reply)
                reply = []
                n_reply += 1
        print(n_reply)
        for i in range(len(sessions)):
            utter = sessions[i]
            reply = replies[i]
            for response in reply:
                for c in utter:
                    f3.write(c + '\n')
                f3.write(response + '\n')
                f3.write("\n")

def get_selection_result(reply_file, result_file, to_file):
    with codecs.open(result_file, 'r', encoding='utf8') as f1, codecs.open(reply_file, 'r', encoding='utf8') as f2, codecs.open(to_file, 'w', encoding='utf8') as f3:
        results = []
        for line in f1:
            results.append(line.strip())
        replies = []
        reply = []
        n_reply = 0
        for line in f2:
            if line != "\n":
                reply.append(line.strip())
            else:
                replies.append(reply)
                if len(reply) != 10:
                    print(reply)
                reply = []
                n_reply += 1
        print(len(replies))
        results_list = []
        n = 0
        for i in range(len(replies)):
            results_list.append(results[n:n+len(replies[i])])
            n += len(replies[i])
        assert len(results_list) == len(replies)
        for i in range(len(results_list)):
            if len(results_list[i]) != len(replies[i]):
                print(i)
        for i in range(len(results_list)):
            f3.write(str(results_list[i].index(max(results_list[i]))) + '\n')


def show_selection_result(context_file, reply_file, gold_file, to_file):
    with codecs.open(context_file, 'r', encoding='utf8') as f1, codecs.open(reply_file, 'r', encoding='utf8') as f2, codecs.open(gold_file, 'r', encoding='utf8') as f3, codecs.open(to_file, 'w', encoding='utf8') as f4:
        n_total_utter = 0
        sessions = []
        utterance = []
        for line in f1:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions.append(utterance)
                utterance = []
                n_total_utter += 1
        print(n_total_utter)
        replies = []
        reply = []
        n_reply = 0
        for line in f2:
            if line != "\n":
                reply.append(line.strip())
            else:
                replies.append(reply)
                if len(reply) != 10:
                    print(reply)
                reply = []
                n_reply += 1
        print(n_reply)
        gold_label = []
        for line in f3:
            gold_label.append(int(line.strip()))
        for i in range(len(sessions)):
            utter = sessions[i]
            for u in utter:
                f4.write(str(u)+'\n')
            reply = replies[i]
            f4.write(str(reply[gold_label[i]]) + '\n')
            f4.write('\n')


def repalce_generate(generate_file, selection_file, new_generate_file, gold_file, gene_file):
    with codecs.open(generate_file, 'r', encoding='utf8') as f1, codecs.open(selection_file, 'r', encoding='utf8') as f2, \
            codecs.open(new_generate_file, 'w', encoding='utf8') as f3, codecs.open(gold_file,'w', encoding='utf8') as f4, codecs.open(gene_file,'w', encoding='utf8') as f5:
        n_total = 0
        g_total_list = []
        sessions = {}
        utterance = []
        n_utterance = 0
        for line in f1:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions["utterance"] = utterance[:n_utterance-1]
                sessions["response"] = utterance[n_utterance-1]
                sessions["label"] = -1
                g_total_list.append(sessions)
                utterance = []
                sessions = {}
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
        print(n_total)

        n_total = 0
        s_total_list = []
        sessions = {}
        utterance = []
        n_utterance = 0
        for line in f2:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions["utterance"] = utterance[:n_utterance - 1]
                sessions["response"] = utterance[n_utterance - 1]
                sessions["label"] = -1
                s_total_list.append(sessions)
                utterance = []
                sessions = {}
                n_utterance = 0
                n_total += 1
                continue
            n_utterance += 1
        print(n_total)
        # first_u_g = []
        # for i in range(len(g_total_list)):
        #     first_u_g.append((g_total_list[i]["utterance"][0],g_total_list[i]["response"]))
        n = 0
        for i in range(len(s_total_list)):
            for j in range(len(g_total_list)):
                if s_total_list[i]["utterance"] == g_total_list[j]["utterance"]:
                    f4.write(s_total_list[i]["response"] + '\n')
                    f5.write(g_total_list[j]["response"] + '\n')
                    g_total_list[j]["response"] = s_total_list[i]["response"]
                    n += 1
        for i in range(len(g_total_list)):
            f3.write(g_total_list[i]["response"] + '\n')
        print(n)


def ensemble(model1, model2, model3, model4, final):
    with codecs.open(model1, 'r', encoding='utf8') as f1, codecs.open(model2, 'r', encoding='utf8') as f2, \
            codecs.open(model3, 'r', encoding='utf8') as f3, codecs.open(model4,'r', encoding='utf8') as f4, codecs.open(final,'w', encoding='utf8') as f5:
        model1_list, model2_list, model3_list, model4_list, final_list = [], [], [], [], []
        for line in f1:
            model1_list.append(float(line.strip()))
        for line in f2:
            model2_list.append(float(line.strip()))
        for line in f3:
            model3_list.append(float(line.strip()))
        for line in f4:
            model4_list.append(float(line.strip()))
        for i in range(len(model1_list)):
            final_val = 0.25 * model1_list[i] + 0.25 * model2_list[i] + 0.25 * model3_list[i] + 0.25 * model4_list[i]
            final_list.append(final_val)
        for val in final_list:
            f5.write(str(val) + "\n")


def p1(gold, pred):
    acc = 0
    pre_list = []
    with open(pred, 'r') as f:
        for line in f:
            pre_list.append(float(line))
    goldlabel = []
    with open(gold, 'r') as f:
        for line in f:
            goldlabel.append(int(line))
    for i in range(len(goldlabel)):
        if pre_list[i] == goldlabel[i]:
            acc += 1
    return acc

def count_diff(file1, file2):
    pre_list = []
    with open(file1, 'r') as f:
        for line in f:
            pre_list.append(float(line))
    goldlabel = []
    with open(file2, 'r') as f:
        for line in f:
            goldlabel.append(float(line))
    n = 0
    for i in range(len(pre_list)):
        if pre_list[i] != goldlabel[i]:
            n += 1
    print(n)

def case_study(file1, file2, file3):
    pre_list_1 = []
    with open(file1, 'r') as f:
        for line in f:
            pre_list_1.append(float(line))
    pre_list_2 = []
    with open(file2, 'r') as f:
        for line in f:
            pre_list_2.append(float(line))
    gold_list = []
    with open(file3, 'r') as f:
        for line in f:
            gold_list.append(float(line))
    assert len(pre_list_1) == len(pre_list_2)
    assert len(pre_list_2) == len(gold_list)
    n1 = 0
    for i in range(len(pre_list_1)):
        if pre_list_1[i] == gold_list[i] and pre_list_2[i] != gold_list[i] :
            print(i)
            n1 += 1
    print(n1)


def case_study(file1, file2, file3):
    pre_list_1 = []
    with open(file1, 'r') as f:
        for line in f:
            pre_list_1.append(float(line))
    pre_list_2 = []
    with open(file2, 'r') as f:
        for line in f:
            pre_list_2.append(float(line))
    gold_list = []
    with open(file3, 'r') as f:
        for line in f:
            gold_list.append(float(line))
    assert len(pre_list_1) == len(pre_list_2)
    assert len(pre_list_2) == len(gold_list)
    n1 = 0
    list_index = []
    for i in range(len(pre_list_1)):
        if pre_list_1[i] == gold_list[i] and pre_list_2[i] != gold_list[i] :
            print(i)
            list_index.append(i)
            n1 += 1
    print(n1)
    return list_index

def case_study_show(file_1, file2, gold_file, to_file, index_list):
    with codecs.open(file_1, 'r', encoding='utf8') as f1, codecs.open(file2, 'r', encoding='utf8') as f2, codecs.open(gold_file, 'r', encoding='utf8') as f3, codecs.open(to_file, 'w', encoding='utf8') as f4:
        n_total_utter = 0
        sessions_1 = []
        utterance = []
        for line in f1:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions_1.append(utterance)
                utterance = []
                n_total_utter += 1
        print(n_total_utter)
        n_total_utter = 0
        sessions_2 = []
        utterance = []
        for line in f2:
            if line != "\n":
                utterance.append(line.strip())
            else:
                sessions_2.append(utterance)
                utterance = []
                n_total_utter += 1
        print(n_total_utter)
        gold_label = []
        for line in f3:
            gold_label.append(int(line.strip()))
        for i in index_list:
            for line in sessions_1[i]:
                f4.write(line + '\n')
            f4.write('\n')
            for line in sessions_2[i]:
                f4.write(line + '\n')
            f4.write('\n')
            f4.write(str(gold_label[i]) + '\n')
            f4.write('\n')
