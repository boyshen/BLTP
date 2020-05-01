# -*- encoding: utf-8 -*-
"""
@file: dict_segmentation.py
@time: 2020/4/9 下午4:23
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 词典预测模型。主要包括：
# 1. 拟合词典：根据提供的训练数据创建词典。 并进行保存
# 2. 评估：读取测试数据，并对数据进行分词，写入文本
# 3. 分词：输入某个句子文本，对其进行分词
# 4. 添加新词：对与某个句子中的词汇没有正确识别，可通过添加新词方式实现。
# 5. 加载模型：对拟合保存的模型进行加载
"""

from module.segmentation.dict.word_dictionary import WordDictionary
from module.core.data_tools import DataTools
from module.core.exception import exception_handling, ParameterError
from module.segmentation.dict.disambiguate import Disambiguate
from module.core.matching import Matching
from module.core.threads import MultiThreading
from module.core.writer import Writer
from module.core.segmentation import Segmentation

MAX_MATCHING = 10
QUEUE_MAX_SIZE = 10240

SLEEP_SEC = 0.5


class DictSegmentation(Segmentation):

    def __init__(self, max_matching=MAX_MATCHING):
        """
        初始化对象
        :param max_matching: <int> 词的最大匹配长度。用于在正向匹配和逆向匹配中对词进行划分
        """
        self.max_matching = max_matching
        self.words = None

        self.word_dictionary = WordDictionary()
        self.disambiguate = Disambiguate()
        self.matching = None

    def fit(self, file, split_lab,
            save_model="DictSegmentation.pickle",
            del_start_str=None,
            del_end_str=None,
            regular_func=None,
            ignore_punctuation=True):
        """
        拟合词典
        :param file: <str> 训练文件
        :param split_lab: <str> 训练文本中对词的划分标记
        :param save_model: <str> 保存模型的文件名
        :param del_start_str: <str> 对于训练数据中的文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: <str> 对于训练数据中的文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: <func> 正则化函数
        :param ignore_punctuation: <bool> 是否忽略训练数据中的标点符号，即是否将标点符号加入到词典中
        :return:
        """
        texts = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)

        self.word_dictionary.fit(texts, split_lab=split_lab, ignore_punctuation=ignore_punctuation)
        self.words = self.word_dictionary.get_dictionary()
        self.matching = Matching(self.words, self.max_matching)

        self.word_dictionary.save(save_model)

        print("word count：", len(self.words))

    def eval(self, file, seg_lab=' ', w_file="test.txt", encoding="utf-8", threads=3, del_start_str=None,
             del_end_str=None, regular_func=None):
        """
        评估。采用线程并非机制，读取测试数据集，并进行分词
        :param file: <str> 测试数据文本
        :param seg_lab: <str> 分词标记。在分词完成之后，需要根据某个字符标记词汇，默认为 " "
        :param w_file: <str> 分词结果写入的文件名
        :param encoding: <str> 分词结果写入文件的编码格式
        :param threads: <int> 启动线程数量
        :param del_start_str: <str> 对于数据中的文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: <str> 对于数据中的文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: <func> 正则化函数
        :return:
        """

        # 处理函数。主要使用正向和逆向匹配规则，然后进行消歧义。
        def handle(data, q):
            result = list()
            for d in data:
                forward_seg = self.matching.forward(d)
                reverse_seg = self.matching.reverse(d)

                value = self.disambiguate.disambiguate([forward_seg, reverse_seg])
                value = seg_lab.join(value)
                result.append(value)
                # 处理完一次结果，向消息队列写入 1
                q.put(1)
            return result

        # 预处理文本
        texts = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)

        # 多线程处理
        multi_thread = MultiThreading(threads=threads)
        results = multi_thread.process(texts, handle)

        # 将结果写入文件
        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, text, seg_lab=' ', del_start_str=None, del_end_str=None, regular_func=None):
        """
        输入某个句子或文本，对其进行分词
        :param text: <str> 句子文本
        :param seg_lab: <str> 分词完成之后需要通过某个字符对其进行标记，默认为 " "
        :param del_start_str: <str> 对于文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: <str> 对于文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: <func> 正则化函数
        :return: <str> 分词文本
        """
        text = DataTools.Preprocess.processing_text(text, del_start_str, del_end_str, regular_func)
        forward_seg = self.matching.forward(text)
        reverse_seg = self.matching.reverse(text)

        result = self.disambiguate.disambiguate([forward_seg, reverse_seg])
        return seg_lab.join(result)

    @exception_handling
    def add_word(self, words, is_save=True, model_file='DictSegmentation.pickle'):
        """
        添加词汇到训练词典中
        :param words: <str, list, tuple> 词汇，可以是字符、列表、元祖
        :param is_save: <bool> 对于新加的词汇，是否保存词典
        :param model_file: <str> 需要保存的模型文件名
        :return:
        """
        if not isinstance(words, (str, list, tuple)):
            raise ParameterError("words parameter type must be is [str, list, tuple]")
        if isinstance(words, (list, tuple)):
            if not isinstance(words[0], str):
                raise ParameterError("words elements type must be is str")

        if isinstance(words, str):
            self.word_dictionary.add_words(words)
        elif isinstance(words, (list, tuple)):
            for word in words:
                self.word_dictionary.add_words(word)

        words = self.word_dictionary.get_dictionary()
        self.matching.update_words(words)

        if is_save:
            self.word_dictionary.save(model_file)

    @staticmethod
    def load(model_file, max_matching=MAX_MATCHING):
        """
        加载词典分词对象
        :param model_file: <str> 保存的词典文件，也叫模型文件
        :param max_matching: <int> 最大匹配词长度，用于正向和逆向匹配的最大匹配词长度
        :return: <DictSegmentation> 词典分词对象
        """
        dict_seg = DictSegmentation(max_matching)
        dict_seg.word_dictionary = WordDictionary.load(model_file)
        dict_seg.words = dict_seg.word_dictionary.get_dictionary()
        dict_seg.matching = Matching(dict_seg.words, max_matching)

        return dict_seg


def test():
    max_matching = 10
    train_file = "../../../data/msr_training_debug.utf8"
    text_file = "../../../data/msr_test_debug.utf8"
    model_file = "../../../model/dict_segmentation_model_debug.pickle"
    save_result = "../../../result/msr_test_result.utf8"

    dict_seg = DictSegmentation(max_matching=max_matching)
    dict_seg.fit(train_file, split_lab=' ', save_model=model_file)
    dict_seg.eval(text_file, seg_lab='  ', w_file=save_result)


def test_load():
    max_matching = 10
    # model_file = "../../../model/dict_segmentation.pickle"
    model_file = "../../../model/dict_segmentation_model_debug.pickle"
    text = "他要与中国人合作。"
    dict_seg = DictSegmentation.load(model_file, max_matching)

    result = dict_seg.cut(text)
    print("predict result : ", result)

    dict_seg.add_word(('中国人', "要与"), is_save=False)
    result = dict_seg.cut(text)
    print("predict result : ", result)

    # text_file = "../../data/icwb2-data/testing/msr_test.utf8"
    # # text_file = "../../data/msr_test_debug3.utf8"
    # save_result = "../../result/msr_test_result.utf8"
    # dict_seg.eval(text_file, seg_lab='  ', threads=5, w_file=save_result)


if __name__ == '__main__':
    test()
    # test_load()
