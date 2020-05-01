# -*- encoding: utf-8 -*-
"""
@file: ngram_disambiguate.py
@time: 2020/4/17 下午4:02
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: N_gram 消歧义。
# 1. 消除歧义。
#    提供 Laplace(拉普拉斯平滑) 和 无平滑技术两种概率计算模式
"""

import numpy as np

from module.core.exception import exception_handling, ParameterError


class NgramDisambiguate(object):
    # 无平滑
    # No_Smoothing = "no_smoothing"
    # 拉普拉斯平滑
    Laplace = "Laplace"

    def __init__(self, ngram_dict: object, smoothing: object = None) -> object:
        # 概率计算方法。目前只有 Laplace、
        self.__smoothing = NgramDisambiguate.Laplace if smoothing is None else smoothing
        # ngram 词典。对词频、n阶词频、单词数量的统计
        self.__ngram_dict = ngram_dict
        # N_gram 模式，二元或三元模式
        self.__n = ngram_dict.get_n()
        # 总词数。从 ngram 词典中获取。
        self.__total_words_num = ngram_dict.get_total_words_num()

    def __get_word_prob(self, word):
        """
        获取某个单词的概率。使用 log 计算。log(词频／总词数)
        :param word: <str> 单词
        :return:<float> 概率值
        """
        word_freq = self.__ngram_dict.get_word_freq(word)
        word_prob = np.log(word_freq / self.__total_words_num)
        return word_prob

    def __get_n_word_prob(self, words):
        """
        获取n阶词频概率。主要使用 Laplace 模式和无模式两种方式。
        no_smoothing 模式: log(n阶词频/词频)
        Laplace 模式：log((n阶词频 + 1)/(词频 + V))。 V 表示总词数
        :param words: <list, tuple> 词汇表。要求格式为list,tuple 两种。
        例如句子："hello word" 则输入格式为 ['word', 'hello']。即表中第一个元素为在文本中索引为 n，第二个元素为 n-1，第三个为 n-2
        :return: <float> 概率值
        """
        # 获取n阶词频
        n_word_freq = self.__ngram_dict.get_n_word_freq(words)

        # 获取单个词词频
        word_freq = 0
        for i in range(1, len(words)):
            word_freq += self.__ngram_dict.get_word_freq(words[i])

        # no_smoothing 模式下计算概率
        # if self.__smoothing == NgramDisambiguate.No_Smoothing:
        #     return np.log(n_word_freq / word_freq)

        # Laplace 模式下计算概率
        if self.__smoothing == NgramDisambiguate.Laplace:
            return np.log((n_word_freq + 1) / (word_freq + self.__total_words_num))

    def __get_score(self, texts, split_lab=None, print_prob=False):
        """
        获取评分。通过计算联合概率的方式计算评分。
        :param texts: <list, tuple, str> 文本
        :param split_lab: <str>
        :return: <list<dict>> 列表字典对象。格式如：[{'text':'hello word', 'score':0.9988}]
        """
        score = list()
        for text in texts:
            # 如果是字符文本，先分词
            if isinstance(text, str):
                text = text.strip().split(split_lab)

            # 跳过空格字符。保存新的词汇
            new_text = list()
            for word in text:
                if word == '' or word == " " or len(word) == 0:
                    continue
                new_text.append(word)

            # 计算联合概率值
            # 获取前n-1、n-2单个词频概率
            word_prob = [self.__get_word_prob(new_text[i]) for i in range(self.__n - 1)]
            # 计算 n、n-1、或 n、n-1、n-2 的同时出现的概率
            for i in range(self.__n - 1, len(new_text)):
                n_words = []
                for j in range(self.__n):
                    n_words.append(new_text[i - j])
                word_prob.append(self.__get_n_word_prob(n_words))

            # 如果需要打印联合概率，则在此输出
            if print_prob:
                print("smoothing: {}".format(self.__smoothing))
                print("text: {}".format(text))
                print("prob: {}".format(word_prob))
                print("score: {}".format(np.sum(word_prob)))
                print()

            # 评分。将联合概率相加。
            score.append(np.sum(word_prob))

        return score

    @exception_handling
    def disambiguate(self, texts, split_lab=None, print_prob=False, need_score=False):
        """
        N_gram 消歧义.
        :param texts: <list, tuple> 文本。文本需要是list或tuple两种类型。
        其中元素可以(list,tuple,str) 。如果元素是 str 类型，需要提供 split_labs。如 ["hello word",...] 或 [["hello", "world"]]
        :param split_lab: <str> 分割符。如果提供的元素是字符。如 "hello world" 需要提供分割符为 " "
        :param print_prob: <bool> 输出概率。即每个词频，n阶词频的计算概率
        :param need_score: <bool>  是否返回评分。
        :return: <str, float> 。如果 need_score = True 返回文本和评分。否则返回文本
        """

        if not isinstance(texts, (list, tuple)):
            raise ParameterError(
                "Input texts parameter type must be is list or tuple, but actually get {}".format(type(texts)))
        for text in texts:
            if not isinstance(text, (str, list, tuple)):
                raise ParameterError(
                    "Input texts elements type must is [str、list、tuple], but actually get {}".format(type(text)))
            if isinstance(text, str):
                if split_lab is None:
                    raise ParameterError("Input texts elements type is str, must provide split label ")

        score = self.__get_score(texts, split_lab, print_prob)
        # 选择最大评分的文本
        max_score_arg = np.array(score).argmax()
        if need_score:
            return texts[max_score_arg], score[max_score_arg]

        return texts[max_score_arg]


def test():
    from module.ngram_seg.ngram_dictionary import NgramDictionary
    texts = ["心  静  渐  知  春  似  海  ，  花  深  每  觉  影  生  香  。",
             "心  静  渐  知  春  似  海  ，  花  深  每  觉  影  生  香  。"]

    n = 2
    # n = 3
    word_dict = NgramDictionary(n=n)
    word_dict.fit(texts, split_lab="  ")

    laplace = NgramDisambiguate(word_dict)

    disambiguate_text = [[NgramDictionary.START_TAG, "心", "静", "渐", "知", NgramDictionary.END_TAG],
                         [NgramDictionary.START_TAG, "心静", "渐知", NgramDictionary.END_TAG]]

    text, score = laplace.disambiguate(disambiguate_text, split_lab="  ", print_prob=True, need_score=True)

    print("best text:", text)
    print("score: ", score)


if __name__ == "__main__":
    test()
