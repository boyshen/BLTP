# -*- encoding: utf-8 -*-
"""
@file: disambiguate.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 消歧规则，定义三种消歧义的规则，分别是最大平均词长度消歧义、最小词变化率、单个词频对数之和
#
"""
import math
import random

from module.core.exception import exception_handling, ParameterError


class Disambiguate(object):

    def __init__(self):
        self.__text = "text"
        self.__score = "score"

        self.__max_avg_word_length = "max_avg_word_length"
        self.__min_change_word_rate = "min_change_word_rate"
        self.__sum_log_word_freq = "sum_log_word_freq"
        self.__disambiguate_type = [self.__max_avg_word_length,
                                    self.__min_change_word_rate,
                                    self.__sum_log_word_freq]

    @exception_handling
    def __check(self, texts, disambiguate_type, lab=None):
        """
        检查函数。检查输入的参数是否符合要求
        :param texts: <list, tuple> 文本
        :param disambiguate_type: <str> 消歧类型
        :param lab: <str> 分词标记
        :return:
        """
        if disambiguate_type not in self.__disambiguate_type:
            raise ParameterError(
                "disambiguate_type must be is {}, but actually get {}".format(self.__disambiguate_type,
                                                                              disambiguate_type))

        if not isinstance(texts, (list, tuple)):
            raise ParameterError("texts parameter must be is [list,tuple], bug actually get{}".format(type(texts)))

        if len(texts) == 0:
            raise ParameterError("texts list or tuple element cannot be empty ")

        for text in texts:
            if not isinstance(text, (list, tuple, str)):
                raise ParameterError(
                    "texts element type must be is [list,tuple,str], but actually get {}".format(type(text)))
            if isinstance(text, str) and lab is None:
                raise ParameterError("texts elements of character type must have lab parameter")

    def __get_score(self, texts, disambiguate_type, lab=None):
        """
        根据不同消歧类型，对文本进行评分
        :param texts: <list, tuple> 文本
        :param lab: <str> 分词标记
        :param disambiguate_type: <str> 消歧类型
        :return: <list> 文本评分
        """
        self.__check(texts, disambiguate_type, lab)

        text_score = list()
        for text in texts:

            # 如果输入的列表元素是字符串，则根据提供的划分标签split_lab进行分词
            # 例如 "生活／水平" 根据 "／" 进行分词。
            # 得到结果为 ["生活","水平"]
            if isinstance(text, str):
                new_text = list()
                words = text.strip().split(lab)
                for word in words:
                    if word == '' or word == " " or len(word) == 0 or word == lab:
                        continue
                    new_text.append(word)
                text = new_text

            # 根据消歧类型对句子进行评分
            score = 0

            # 最大平均词长度。即一个文本中的总字数与总词汇数的比值
            if disambiguate_type == self.__max_avg_word_length:
                text_length = len(''.join(text))
                words_length = len(text)
                score = float(text_length) / words_length

            # 词语最小变化率。使用标准差来计算。
            # 均值avg = 总字数除以总词汇量。
            # xi = 每个词的字数。i = {1，2，3，4...}
            # 标准差 = sqrt(((xi - avg )^2) / N) i = {1,2,3,4} 即词汇下标，N 为总词汇数量
            if disambiguate_type == self.__min_change_word_rate:
                n = len(text)
                avg = len(''.join(text)) / n
                value = 0
                for word in text:
                    value = value + math.pow(len(word) - avg, 2)
                score = math.sqrt(value / n)

            # 所有单个字的出现频率的对数和
            # 先统计文本中单个字出现的次数，然后对次数进行log运算，最后相加得出评分
            if disambiguate_type == self.__sum_log_word_freq:
                counter = dict()
                for word in text:
                    if len(word) == 1:
                        if word in list(counter.keys()):
                            counter[word] = counter[word] + 1
                        else:
                            counter[word] = 1

                score = 0
                for _, freq in counter.items():
                    score = score + math.log(freq)

            # 保存对每个文本的评分
            text_score.append(score)

        return text_score

    def __get_result(self, texts, text_score, disambiguate_type):
        """
        根据评分选取文本
        :param texts: <list, tuple> 文本
        :param text_score: <list> 文本评分
        :param disambiguate_type: <str> 消歧类型
        :return: 选取结果
        """

        result = list()

        # 根据不同的消歧义算法选择最大，最小评分值
        value = 0
        if disambiguate_type == self.__max_avg_word_length or \
                disambiguate_type == self.__sum_log_word_freq:
            value = max(text_score)
        if disambiguate_type == self.__min_change_word_rate:
            value = min(text_score)

        # 根据评分值获取到文本。文本可能不止一个。
        for (text, score) in zip(texts, text_score):
            if score == value:
                result.append({self.__text: text, self.__score: score})

        return result

    def max_avg_word_length(self, texts, lab=None):
        """
        最大平均词长度消歧义
        :param texts: <list, tuple> 文本。文本格式可以是[ ["生活","水平"] ] 或 ["生活/水平"] 两种格式，
        如果是后者需要提供 lab 即分词标记。如"/" 。
        :param lab: <str> 分词标记，例如 ["生活/水平"] 分词标记为 "／"
        :return: <list<dict>> 列表字典对象，字典 {'text': text, 'score': score} ，即每个文本对应各自评分
        """
        text_score = self.__get_score(texts, self.__max_avg_word_length, lab)

        result = self.__get_result(texts, text_score, self.__max_avg_word_length)

        return result

    def min_change_word_rate(self, texts, lab=None):
        """
        最小词变化率
        :param texts:<list, tuple> 文本
        :param lab: <str> 分词标记
        :return:list<dict> 列表字典对象，字典 {'text': text, 'score': score}
        """
        text_score = self.__get_score(texts, self.__min_change_word_rate, lab)

        result = self.__get_result(texts, text_score, self.__min_change_word_rate)

        return result

    def sum_log_word_freq(self, texts, lab=None):
        """
        单个词频对数之和
        :param texts: <list, tuple> 文本
        :param lab: <str> 分词标记
        :return: list<dict> 列表字典对象，字典 {'text': text, 'score': score}
        """
        text_score = self.__get_score(texts, self.__sum_log_word_freq, lab)

        result = self.__get_result(texts, text_score, self.__sum_log_word_freq)

        return result

    def disambiguate(self, texts, lab=None, need_score=False):
        """
        消除歧义，结合最大平均词长度、最小词变化率、单个词频对数之和 三种消歧方法进行。
        消歧流程中，如果任一种方法只返回1个结果，则结束返回。如果三种方法返回的结果都超过1个，
        则进行匹配和随机抽取结果。
        :param texts: <list, tuple> 文本
        :param lab: <str> 分词标记
        :param need_score: <bool> 是否需要返回评分
        :return: <str, list> 如果 need_score = True ,
        则返回类似这样的结果 [{'text': '生活/水平', 'score': 2.0, 'rule': 'compare'}]
        否则返回字符串，如 '生活/水平'
        """

        def output(rule):
            if need_score:
                result[0]['rule'] = rule
                return result
            else:
                return result[0][self.__text]

        all_result = list()
        result = self.min_change_word_rate(texts, lab)
        if len(result) == 1:
            return output(self.__min_change_word_rate)

        all_result = all_result + result
        result = self.sum_log_word_freq(texts, lab)
        if len(result) == 1:
            return output(self.__sum_log_word_freq)

        all_result = all_result + result
        result = self.max_avg_word_length(texts, lab)
        if len(result) == 1:
            return output(self.__max_avg_word_length)

        # 从所有结果种抽取评分最大的文本
        all_result = all_result + result
        score = 0
        for value in all_result:
            if value[self.__score] > score:
                result = value
                score = value[self.__score]

        if score != 0:
            result = [result]
            return output("compare")

        # 随机选择某个文本进行返回
        index = random.randint(0, len(all_result) - 1)
        result = all_result[index]
        result = [result]

        return output("random")


def test():
    text = [("生活", "水平"), ("生", "活", "水平"), ("生", "活水", "平"), ("水平", "生活")]
    # text = ["中国/人民/万岁", "中国人/民/万岁"]
    # text = ["人/与/自然，人/与/环境", "人/与/自然，人与/环/境"]
    disambiguate = Disambiguate()

    # value = disambiguate.max_avg_word_length(text, lab='/')
    value = disambiguate.max_avg_word_length(text)
    print("max avg word length : ", value)

    # value = disambiguate.min_change_word_rate(text, lab='/')
    value = disambiguate.min_change_word_rate(text)
    print("min change word rate : ", value)

    # value = disambiguate.sum_log_word_freq(text, lab='/')
    value = disambiguate.sum_log_word_freq(text)
    print("sum log word freq : ", value)

    # value = disambiguate.disambiguate(text, lab='/', need_score=True)
    value = disambiguate.disambiguate(text)
    print("value: ", value)


if __name__ == "__main__":
    test()
