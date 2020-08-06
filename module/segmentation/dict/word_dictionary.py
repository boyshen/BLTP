"""
@file: word_dictionary.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 继承 dictionary.Dictionary 类，实现创建词表、拟合词表、添加新词、获取词表函数
"""

from tqdm import tqdm
from collections import Counter
from module.core.dictionary import Dictionary
from module.core.exception import exception_handling, NotFitException


class WordDictionary(Dictionary):
    """
    词典类，继承 Dictionary 类，用于创建词典列表。
    """

    def __init__(self):
        super(WordDictionary, self).__init__()
        self.__words = list()

        self.is_fit = False

    def fit(self, dataset):
        """
        创集词典列表
        :param dataset: (list of list, mandatory) 列表数据集，通常是预处理之后的文本
        :return:
        """

        word_counter = Counter()
        for data in tqdm(dataset):
            word_counter.update(data)

        for word, _ in word_counter.items():
            self.__words.append(word)

        self.is_fit = True

    @exception_handling
    def get_dictionary(self):
        """
        获取fit之后的词典，返回元组类型
        :return:
        """
        if not self.is_fit:
            raise NotFitException(self.__class__.__name__)

        return tuple(self.__words)

    def add_words(self, words):
        """
        添加新的词汇。将新的词汇添加到词典中
        :param words: (int or str or list or tuple, mandatory) 词汇，可以是列表、元祖、字符、数字类型
        :return:
        """
        if isinstance(words, (str, int)):
            if words != '' and words != " " and len(words) != 0:
                if words not in self.__words:
                    self.__words.append(words)

        if isinstance(words, (list, tuple)):
            for word in words:
                if word == '' or word == " " or len(word) == 0:
                    continue

                if word not in self.__words:
                    self.__words.extend(word)


def test_module_func():
    dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '民族', '的', '感情', '问题', '。']]
    word_num = 10

    word_dictionary = WordDictionary()
    word_dictionary.fit(dataset)

    words = word_dictionary.get_dictionary()

    assert len(words) == word_num, \
        "{} Error: fit get words num:{}, expect get: {}".format(word_dictionary.fit.__name__, len(words), word_num)

    print("words: ", words)
    print("words number: ", len(words))

    word_dictionary.add_words(["1", "2", "3"])
    word_dictionary.add_words("5")
    words = word_dictionary.get_dictionary()

    assert len(words) == word_num + 4, \
        "{} Error: add get words num:{}, expect get{}".format(word_dictionary.add_words.__name__, len(words),
                                                              word_num + 4)

    words = word_dictionary.get_dictionary()
    print("words: ", words)
    print("words number: ", len(words))

    word_dictionary.save("./word_dictionary.pickle")

    load_word_dictionary = WordDictionary.load("./word_dictionary.pickle")
    assert len(load_word_dictionary.get_dictionary()) == word_num + 4, \
        "{} Error: load get words num:{}, expect get{}".format(word_dictionary.load.__name__, len(words), word_num + 4)

    print("test ok!")


if __name__ == "__main__":
    test_module_func()
