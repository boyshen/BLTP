# -*- encoding: utf-8 -*-
"""
@file: hmm_dictionary.py
@time: 2020/4/22 下午5:09
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 1. fit 拟合。输入训练数据。创建发射矩阵和状态转移矩阵
# 2. get_launch_prob 获取发射概率。输入字和当前状态，获取该字的发射概率。
# 3. encoding 编码。将采用 " " 或 "／" 等标记分词的数据集转换成"BMES"的形式
# 4. decoding 加码。将采用 "BMES" 标记的文本转换成词组
# 5. get_state_transition_prob 获取状态转移概率。字典格式，下一个状态和概率值。例如:{'B':0.12, 'E':0.28, 'S':'0.6', 'M':0.0}
"""

import numpy as np

from tqdm import tqdm
from collections import Counter
from module.core.dictionary import Dictionary
from module.core.exception import exception_handling, ParameterError


class HmmDictionary(Dictionary):
    MINIMUM_PROP = 3.14e-100
    MAXIMUM_PROP = 0.9

    STATE_B = "B"
    STATE_M = "M"
    STATE_E = "E"
    STATE_S = "S"
    STATE_SEQUENCE = (STATE_B, STATE_M, STATE_E, STATE_S)

    def __init__(self):
        super(HmmDictionary, self).__init__()

        # 用于保存每个字，每个字符对应一个唯一的 token 值
        self.word_token = {}

        # 用于保存每个状态值，每个状态值对应一个唯一的 token 值
        self.state_token = {}
        for state in HmmDictionary.STATE_SEQUENCE:
            self.state_token[state] = len(self.state_token)
        self.state_token[self.START_TAG] = len(self.state_token)
        self.state_token[self.END_TAG] = len(self.state_token)

        # 发射概率矩阵
        self.__launch_matrix = None

        # 状态转移矩阵。初始化状态转移矩阵，将矩阵元素设置为极小值=3.14e-100 。 矩阵大小为[5, 6]
        self.__state_transition_matrix = np.zeros([len(self.state_token) - 1, len(self.state_token)])

    @staticmethod
    def encoding(texts):
        """
        初始化数据及标签。将采用 " " 或 "／" 等标记分词的数据集转换成"BMES"的形式。
        例如输入： [["小明", "是", "中国人"]] ,返回 ['小','明','是','中','国','人'], ['B','E','S','B','M','E']
        :param texts: (list of list, mandatory) 文本数据集。
        :return: data (list), label: (list) 返回数据列表和标签列表
        """

        def get_label(w):
            """
            输入词或字转换成"BMES"
            :param w: (str) 字或词
            :return: (str) 标签字符
            """
            if len(w) > 2:
                word_lab = HmmDictionary.STATE_B + len(w[1:-1]) * HmmDictionary.STATE_M + HmmDictionary.STATE_E
            elif len(w) == 2:
                word_lab = HmmDictionary.STATE_B + HmmDictionary.STATE_E
            else:
                word_lab = HmmDictionary.STATE_S
            return word_lab

        data, label = list(), list()
        for text in tqdm(texts):
            # text = text.strip().split(split_lab)
            sent, lab = "", ""
            for word in text:
                if word == '' or word == " " or len(word) == 0:
                    continue
                sent = sent + word
                lab = lab + get_label(word)

            data.append(list(sent))
            label.append(list(lab))

        return data, label

    @staticmethod
    @exception_handling
    def decoding(text, label):
        """
        解码. 将文本列表 ['小','明','是','中','国','人'] 和 ['B','E','S','B','M','E'] 转换成 ["小明","是","中国人"]
        :param text: (list, mandatory) 文本列表。
        :param label: (list, mandatory) 文本列表对应的标签。要求 text 和 label 的长度必须一致
        :return: (list) 解码后的文本列表
        """

        if len(text) != len(label):
            raise ParameterError("Parameter text={} and label={} length must be equal".format(len(text), len(label)))

        words = list()
        string = ''
        for word, lab in zip(text, label):
            if lab == HmmDictionary.STATE_S:
                words.append(word)
            elif lab == HmmDictionary.STATE_B and string == "":
                string = word
            elif lab == HmmDictionary.STATE_M and string != "":
                string += word
            elif lab == HmmDictionary.STATE_E and string != "":
                string += word
                words.append(string)
                string = ""

        return words

    def __init_word_token(self, data):
        """
        初始化 word_token . 为每个字分配一个唯一标识符 token
        :param data: (list, mandatory) 数据列表。格式：[['小','明','是','中','国','人']...]
        :return:
        """
        word_count = Counter()
        for d in tqdm(data):
            word_count.update(d)
        for word, _ in tqdm(word_count.items()):
            self.word_token[word] = len(self.word_token)

    def __init_launch_matrix(self, data, label):
        """
        初始化发射矩阵
        :param data: (list, mandatory) 数据列表 例如：[['小','明','是','中','国','人']...]
        :param label: (label, mandatory) 数据对应的标签 例如 [['B','E','S','B','M','E']]
        :return: 
        """""
        # 初始化发射矩阵大小。矩阵大小为 [字数, 状态序列数=4]
        self.__launch_matrix = np.zeros([len(self.word_token), len(HmmDictionary.STATE_SEQUENCE)])

        # 统计每个字在 "BMES" 各个状态下的频次
        for d, lab in tqdm(zip(data, label)):
            for word, state in zip(d, lab):
                self.__launch_matrix[self.word_token[word]][self.state_token[state]] += 1

        # 归一化。对每列状态值进行归一化处理
        self.__launch_matrix = self.__launch_matrix / self.__launch_matrix.sum(axis=0)

        # 对为 0.0 的值赋予极小值=3.14e-100 。
        self.__launch_matrix[self.__launch_matrix == 0.0] = HmmDictionary.MINIMUM_PROP

    def __init_state_transition_matrix(self, label):
        """
        初始化状态转移矩阵
        :param label: (label) 数据对应的标签 例如 [['B','E','S','B','M','E']]
        :return:
        """

        for lab in tqdm(label):
            # 添加开始和结束标记
            lab = [self.START_TAG] + lab + [self.END_TAG]

            # 循环添加每个状态。
            # 例如 lab = ['(Start)','B','M','E','(End)']。
            # lab[:-1] = ['(Start)','B','M','E'] 当前状态
            # lab[1:] = ['B','M','E','(End)'] 下一个状态
            for state, next_state in zip(lab[:-1], lab[1:]):
                self.__state_transition_matrix[self.state_token[state]][self.state_token[next_state]] += 1

        # 归一化。对每行的状态值进行归一化处理
        state_sum = self.__state_transition_matrix.sum(axis=1).reshape(-1, 1)
        self.__state_transition_matrix = self.__state_transition_matrix / state_sum

        # 对为0的值赋予极小值 3.14e-100
        # self.__state_transition_matrix[self.__state_transition_matrix == 0.0] = HmmDictionary.MINIMUM_PROP

    def fit(self, dataset):
        """
        拟合
        :param dataset: (list, mandatory) 文本数据集。例如 ["小明 是 中国人",...]
        :return:
        """
        # 初始化数据标签
        data, label = HmmDictionary.encoding(dataset)
        # 初始化 word_token
        self.__init_word_token(data)
        # 初始化发射矩阵
        self.__init_launch_matrix(data, label)
        # 初始化转移矩阵
        self.__init_state_transition_matrix(label)

        print("word number: ", len(self.word_token))
        # print("launch: ", self.__launch_matrix)
        # print("transition: ", self.__state_transition_matrix)

    def get_launch_prob(self, word, state):
        """
        获取发射概率
        :param word: (str, mandatory) 中文字
        :param state: (str, mandatory) 中文字对应的状态值。"BMES" 中的任一个
        :return: (float) 成功：返回概率值，失败：判断状态, 如果：state='S' , 返回极大值=0.9，否则返回极小值 3.14e-100
        """
        try:
            w_token = self.word_token[word]
        except KeyError:
            if state == HmmDictionary.STATE_S:
                return HmmDictionary.MAXIMUM_PROP

            return HmmDictionary.MINIMUM_PROP

        s_token = self.state_token[state]
        return self.__launch_matrix[w_token][s_token]

    @exception_handling
    def get_state_transition_prob(self, state, is_last_state=False):
        """
        获取状态转移概率。从当前状态到下一个状态的概率值
        :param state: (str, mandatory) 当前状态值
        :param is_last_state: (bool, optional, default=False) 是否是最后一个状态值。如果为 True 则返回当前状态到 (End) 概率
        :return: (dict) 下一个状态和概率值。例如:{'B':0.12, 'E':0.28, 'S':'0.6', 'M':0.0}
        """
        if state == self.END_TAG:
            raise ParameterError("Current state value cannot is {}".format(state), level=ParameterError.warning)

        result = dict()

        if is_last_state:
            value = self.__state_transition_matrix[self.state_token[state]][self.state_token[self.END_TAG]]
            if value != 0.0:
                result[self.END_TAG] = value
        else:
            state_seq = HmmDictionary.STATE_SEQUENCE
            for next_state in state_seq:
                value = self.__state_transition_matrix[self.state_token[state]][self.state_token[next_state]]
                if value == 0.0:
                    continue
                result[next_state] = value

        return result


def test_module_func():
    dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '民族', '的', '感情', '问题', '。'],
               ['小明', '是', '中国人'],
               ['小红', '爱', '中国']]

    hmm_dict = HmmDictionary()
    hmm_dict.fit(dataset)

    launch_prob = hmm_dict.get_launch_prob('民', 'B')
    assert launch_prob == 2 / 10, \
        "{} Error: launch prob expect get:{}, but actually get:{}".format(hmm_dict.get_launch_prob.__name__, 2 / 10,
                                                                          launch_prob)
    trains_prob = hmm_dict.get_state_transition_prob(Dictionary.START_TAG)
    print("<start> next state prob: {}".format(trains_prob))
    expect_value = {'B': 2/3, 'S': 1/3}
    assert trains_prob == expect_value, \
        "{} Error: transition prob expect get:{}, actually get:{} ".format(hmm_dict.get_state_transition_prob.__name__,
                                                                           expect_value, trains_prob)

    text_list = ['小', '明', '是', '中', '国', '人']
    label = ['B', 'E', 'S', 'B', 'M', 'E']
    result = hmm_dict.decoding(text_list, label)
    print("decoding: input text: {}, label: {}, output: {}".format(text_list, label, result))

    print("trans prob: 'E' is last state, next is (END) prob: {}".format(
        hmm_dict.get_state_transition_prob(HmmDictionary.STATE_E, is_last_state=True)))

    print("word dict: ", hmm_dict.word_token)


if __name__ == "__main__":
    test_module_func()
