# -*- encoding: utf-8 -*-
"""
@file: ngram_dictionary.py
@time: 2020/4/15 下午9:09
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: ngram分词词典。主要包括：
# 1. 拟合词典。输入数据集，创建词典、统计词数量、单个词频、n阶词频。n = {2, 3}
# 2. 获取词表。
# 3. 获取词频。
# 4. 获取n阶词频。
"""
import numpy as np
from collections import Counter
from tqdm import tqdm

from module.core.dictionary import Dictionary
from module.core.exception import exception_handling, ParameterError

TWO_TAG = 2
THREE_TAG = 3
N_GRAM = [TWO_TAG, THREE_TAG]


class NgramDictionary(Dictionary):

    @exception_handling
    def __init__(self, n=2):
        super(NgramDictionary, self).__init__()
        if n not in N_GRAM:
            raise ParameterError("parameter n must be within {}".format(N_GRAM))
        self.__n = n

        # 词表。统计文本中的所有出现的单词
        self.__words = [self.UNK_TAG]
        # 词典。统计文本中的所有出现的单词，并给予 token 。token 值可以看作是单词的唯一标识符。
        self.__word_token = {self.UNK_TAG: self.UNK}
        # 单词数量。记录文本中出现的单词。不包括重复单词
        self.__word_num = len(self.__words)
        # 总词数。记录文本中所有单词。包括重复单词
        self.__total_words = len(self.__words)
        # 词频。统计文本中单词频率
        self.__word_freq = Counter()
        # n阶词频。
        self.__n_word_freq = None

    def word_to_token(self, word):
        """
        根据单词获取token值。
        :param word: <str> 单词
        :return: <int> 成功则返回 token， 失败则返回 UNK = 0
        """
        try:
            token = self.__word_token[word]
        except KeyError:
            token = self.UNK

        return token

    def __init_word_freq(self, texts, split_lab):
        """
        初始化词频
        :param texts: <list> 文本数量
        :param split_lab: <str> 划分词频标签
        :return: <list> 返回处理之后的分词文本
        """
        # 遍历文本，统计词频。同时对每个句子添加 <START> 和 <END> 字符。
        # 使用 new_text 保存已经分词和添加 <START> 和 <END> 字符的文本。
        # new_text 是二维列表：[['<START>','文本','<END>'],['<START>'...'<END>']]
        new_text = list()
        for text in tqdm(texts):
            text = text.strip()

            if text == '' or text == ' ' or len(text) == 0:
                continue

            # 添加 <START> 和 <END>
            text = self.START_TAG + split_lab + text + split_lab + self.END_TAG

            # 根据split_lab 进行分词。跳过空格字符
            words = text.split(split_lab)
            new_words = list()
            for word in words:
                if word == '' or word == ' ' or len(word) == 0:
                    continue
                new_words.append(word)

            # 统计词频
            self.__word_freq.update(new_words)
            new_text.append(new_words)
        return new_text

    def __init_words_token(self):
        """
        初始化词表和word_token
        :return:
        """
        # 初始化词表、word_token、total_words
        for word, freq in tqdm(self.__word_freq.items()):
            self.__words.append(word)
            self.__word_token[word] = len(self.__word_token)
            self.__total_words += freq

        # 记录单词数量
        self.__word_num = len(self.__words)

    def __init_n_word_freq(self, texts):
        """
        初始化n阶词频矩阵
        :param texts:<list> 文本列表
        :return:
        """
        # 初始化n阶词频矩阵，如果 n = 2 ，则表示二维矩阵，n = 3 则为三维矩阵。矩阵行和列代表单词的 token、初始化先将其设置为 0
        # 例如 n = 2 。单词 'hello' 和 'word' 两个词矩阵为 [[1,2],[3,4]] 。
        # 词矩阵：
        #   1 为 'hello' 单词前面为 'hello' 词的频次。2 为 'hello' 词前一个是 'word' 的频次。
        #   3 为'word' 词前一个是 'hello' 的频次。4 为 'word'词前一个是 'word' 的频次
        matrix_shape = [self.__word_num] * self.__n
        self.__n_word_freq = np.zeros(matrix_shape, dtype=np.int)
        # self.__n_word_freq = matrix_array.tolist()

        # 统计n阶词频
        for text in tqdm(texts):
            if len(text) > self.__n:
                # 统计n阶词频。n 代表单词的个数。即 n = 2 时候。当前单词与前一个单词出现的次数。通过遍历统计前一个与当前词出现的次数
                # 这里 n_gram = 1 时表示前一个词，n_gram = 2 时候表示前一个词的前一个词
                n_gram = self.__n - 1
                for i in range(len(text) - n_gram):
                    w_t = self.word_to_token(text[i + n_gram])
                    one_gram_w_t = self.word_to_token(text[i + n_gram - 1])

                    # TWO_TAG = 2 表示二元语法。即当前词与前一词有关
                    if self.__n == TWO_TAG:
                        # 统计词频。即发现一次，在其基础上 + 1
                        self.__n_word_freq[w_t][one_gram_w_t] += 1

                    # THREE_TAG = 3 表示三元语法。即当前词与前一个词、和前一词的前一个词有关
                    elif self.__n == THREE_TAG:
                        two_gram_w_t = self.word_to_token(text[i + n_gram - 2])
                        self.__n_word_freq[w_t][one_gram_w_t][two_gram_w_t] += 1

            elif len(text) < self.__n:
                continue

    def fit(self, texts, split_lab):
        """
        拟合操作，初始化词频、n阶词频、词表、单词数量。
        :param texts: <list> 文本
        :param split_lab: <str> 分词标记
        :return:
        """
        new_texts = self.__init_word_freq(texts, split_lab)
        self.__init_words_token()
        self.__init_n_word_freq(new_texts)

        print("word number: ", self.__word_num, "total words number:", self.__total_words)

    def get_words(self):
        """
        获取词表
        :return: <list> 单词统计表
        """
        return self.__words

    def get_word_freq(self, word):
        """
        输入单词获取词频。如果查询到单词，则返回对应的词频，否则返回  0
        :param word: <str> 单词
        :return: <int> 词频
        """
        try:
            w_freq = self.__word_freq[word]
        except KeyError:
            w_freq = 0
        return w_freq

    @exception_handling
    def get_n_word_freq(self, words):
        """
        获取 n 阶词频
        :param words: <list, tuple> 输入的类型必须是list或tuple，同时词数量必须等于 n(n元语法，n=2或n=3).
        同时规定列表中第一个单词为当前词，第二个单词为当前词的前一个，依次类推。例如"A"、"B"、"C" 这三个词。当前词为 'B'，'B' 的上一个词为"A"
        :return: <int> 词频
        """
        if not isinstance(words, (list, tuple)):
            raise ParameterError("input words type must be is [list, tuple], actually get {}".format(type(words)))
        if len(words) != self.__n:
            raise ParameterError("input number of words equal to {}, actually get {}".format(self.__n, len(words)))

        n_word_freq = 0
        if self.__n == TWO_TAG:
            n_word_freq = self.__n_word_freq[self.word_to_token(words[0])][self.word_to_token(words[1])]

        elif self.__n == THREE_TAG:
            w_k = self.word_to_token(words[0])
            one_gram_w_k = self.word_to_token(words[1])
            two_gram_w_k = self.word_to_token(words[2])
            n_word_freq = self.__n_word_freq[w_k][one_gram_w_k][two_gram_w_k]

        return n_word_freq

    def get_n(self):
        """
        获取 n 值。
        :return: <int> self.__n
        """
        return self.__n

    def get_word_num(self):
        """
        获取单词数量
        :return: <int> self.__word_num
        """
        return self.__word_num

    def get_total_words_num(self):
        """
        获取总词数
        :return: <int> self.__total_words
        """
        return self.__total_words


def test():
    texts = ["心  静  渐  知  春  似  海  ，  花  深  每  觉  影  生  香  。",
             "心  静  渐  知  春  似  海  ，  花  深  每  觉  影  生  香  。"]

    # n = 2
    n = 3
    word_dict = NgramDictionary(n=n)
    word_dict.fit(texts, split_lab="  ")

    # 单词 "静" 的词频
    word = "静"
    print("word: {}, freq: {}".format(word, word_dict.get_word_freq(word)))

    # n = 2 时，"静" 的前面是 "心" 的频次
    # words = ["静", "心"]

    # n = 3 时，"渐" 前面是 "静"， "静" 前面是"心" 的频次
    words = ["渐", "静", "心"]
    print("words: {}, n_gram: {}".format(words, word_dict.get_n_word_freq(words)))

    words = word_dict.get_words()
    print("total words: {}, word number: {}".format(words, len(words)))


if __name__ == "__main__":
    test()
