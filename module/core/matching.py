# -*- encoding: utf-8 -*-
"""
@file: matching.py
@time: 2020/4/7 下午6:00
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 词典匹配算法，提供正向匹配和逆向匹配两种模式
"""

try:
    from module.core.exception import exception_handling, \
        NullCharacterException, ParameterError
except ModuleNotFoundError:
    from .exception import exception_handling, \
        NullCharacterException, ParameterError


class Matching(object):

    def __init__(self, dictionary, max_matching=10):
        self.__dictionary = dictionary
        self.__max_matching = max_matching

        self.__forward_type = "forward"
        self.__reverse_type = "reverse"
        self.__matching_type = [self.__forward_type, self.__reverse_type]

    @exception_handling
    def __update(self, text, matching_length, matching_type):
        """
        匹配算法中迭代更新文本和匹配长度
        :param text: (str, mandatory) 当前文本
        :param matching_length: (int, mandatory) 当前匹配长度
        :param matching_type: (str, mandatory) 匹配算法的类型，用于判断截取新文本
        :return: (str and int) 新的文本和匹配长度
        """
        if matching_type not in self.__matching_type:
            raise ParameterError("matching type {}".format(self.__matching_type))

        new_text = None
        # 截取新的文本，将剩余的文本再次进行匹配
        if matching_type == self.__forward_type:
            new_text = text[matching_length:]

        elif matching_type == self.__reverse_type:
            new_text = text[:-matching_length]

        # 如果新的文本存在空格字符，则去除
        new_text = new_text.strip()

        # 更新最大匹配长度
        # 如果剩余文本的字符长度小于默认设置，则以剩余文本字符长度为准
        # 否则以最大匹配字符长度为准
        new_matching_length = len(new_text) if len(new_text) < self.__max_matching else self.__max_matching

        return new_text, new_matching_length

    @exception_handling
    def __matching(self, text, matching_type):
        """
        匹配算法
        :param text: (str, mandatory) 匹配文本
        :param matching_type: (str, mandatory) 类型，forward 和 reverse 两种类型
        :return: (tuple) 分词的元祖数据
        """
        # 检查输出的参数是否有误
        if matching_type not in self.__matching_type:
            raise ParameterError("matching type {}".format(self.__matching_type))

        # 检查输出的文本是否为空，为空则抛出异常
        if len(text) == 0 or text == '' or text == " ":
            raise NullCharacterException("input text cannot be empty")

        new_text = text
        new_matching_length = len(text) if len(text) < self.__max_matching else self.__max_matching

        words = list()

        while True:
            # 如果文本的长度等于 0 ，或者为空字符，则结束
            if new_text == '' or new_text == " " or len(new_text) == 0:
                break

            # 根据定义的最大匹配长度获取单词
            word = ''
            if matching_type is self.__forward_type:
                word = new_text[:new_matching_length]

            elif matching_type is self.__reverse_type:
                word = new_text[-new_matching_length:]

            # 单词匹配，判断单词是否在词表中
            if word in self.__dictionary:
                # 将匹配上的词加入词表
                words.append(word)
                # 更新文本和匹配字符的长度
                new_text, new_matching_length = self.__update(new_text,
                                                              new_matching_length,
                                                              matching_type)

            # 如果只有某个字符，则将该字符作为词汇
            elif len(word) == 1:
                words.append(word)
                new_text, new_matching_length = self.__update(new_text,
                                                              new_matching_length,
                                                              matching_type)

            # 缩短匹配字符的长度
            else:
                new_matching_length = new_matching_length - 1

        return tuple(words)

    def forward(self, text):
        """
        正向匹配
        :param text: (str, mandatory) 文本数据
        :return: (tuple) 分词的数据元祖对象
        """
        words = self.__matching(text, self.__forward_type)
        return words

    def reverse(self, text):
        """
        逆向匹配
        :param text: (str) 文本数据
        :return: (tuple) 分词的数据元祖对象
        """
        words = self.__matching(text, self.__reverse_type)
        new_words = list()

        # 逆向匹配返回的单词是从后往前，需要进行转换
        for i in range(1, len(words) + 1):
            new_words.append(words[-i])

        return tuple(new_words)

    def update_words(self, dictionary):
        """
        更新词典
        :param dictionary: (list, mandatory) 词典
        :return:
        """
        self.__dictionary = dictionary


def test():
    texts = ["他要与中国人合作。", "他来到中国，成为第一个访华的大船主。"]
    dictionary = ["他", "来到", "中国", "成为", "第一个", "访华", "的", "大船主", "要", "中国人", "合作"]

    matching = Matching(dictionary, max_matching=5)
    for text in texts:
        print("text: ", text)
        print("forward matching: ", matching.forward(text))
        print("reverse matching: ", matching.reverse(text))


if __name__ == "__main__":
    test()
