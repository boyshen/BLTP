"""
@file: word_dictionary.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 继承 dictionary.Dictionary 类，实现创建词表、拟合词表、添加新词、获取词表函数
"""

from tqdm import tqdm

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

    def fit(self, texts, split_lab, ignore_punctuation=True):
        """
        创集词典列表
        :param texts: <list> 列表数据集，通常是预处理之后的文本
        :param split_lab: <str> 分词标识符号
        :param ignore_punctuation: <bool> 是否忽略标点符号，即是否将标点符号加入词表。
        :return:
        """

        word_list = list()
        for text in tqdm(texts):
            text = text.strip()

            # 如果文本为空字符，则跳过
            if len(text) == 0:
                continue

            # 忽略文本中的标点符号
            if ignore_punctuation:
                text = self.delete_punctuation(text)

            # 根据分割符划分词汇，剔除重复和空格字符。
            w_list = list(set(text.split(split_lab)))
            new_words = list()
            for w in w_list:
                if w == '' or len(w) == 0 or w == " ":
                    continue
                new_words.append(w)

            word_list = word_list + new_words

        # 去除重复词汇
        word_list = list(set(word_list))
        self.__words = word_list

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
        :param words: <int, str, list, tuple> 词汇，可以是列表、元祖、字符、数字类型
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


def test():
    from module.core.data_tools import DataTools
    file = "../../../data/msr_training_debug.utf8"
    text_data = DataTools.Preprocess.read_file_data(file)

    word_dictionary = WordDictionary()
    word_dictionary.fit(text_data, split_lab="  ", ignore_punctuation=True)
    words = word_dictionary.get_dictionary()

    print("words number: ", len(words))
    print("[0 ~ 1000] words: ", words[:1000])

    word_dictionary.add_words(["1", "2", "3"])
    word_dictionary.add_words("5")
    words = word_dictionary.get_dictionary()
    print("words number: ", len(words))
    print("[0 ~ 1000] words: ", words[:1000])

    word_dictionary.save("/Users/shen/Desktop/me/python/AI/nlp/running/BLTP/model/word_dictionary.pickle")


def test_load():
    word_dictionary = WordDictionary.load(
        "/Users/shen/Desktop/me/python/AI/nlp/running/BLTP/model/word_dictionary.pickle")

    words = word_dictionary.get_dictionary()
    print("words number: ", len(words))


if __name__ == "__main__":
    # test()
    test_load()
