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
        :param max_matching: (int, optional, default=10) 词的最大匹配长度。用于在正向匹配和逆向匹配中对词进行划分
        """
        self.max_matching = max_matching
        self.words = None

        self.word_dictionary = WordDictionary()
        self.disambiguate = Disambiguate()
        self.matching = None

    def fit(self, dataset, save_model="DictSegmentation.pickle"):
        """
        拟合词典
        :param dataset: (list of list, mandatory) 训练数据集
        :param save_model: (str, optional, default=DictSegmentation.pickle) 保存词典
        :return:
        """
        self.word_dictionary.fit(dataset)
        self.words = self.word_dictionary.get_dictionary()
        self.matching = Matching(self.words, self.max_matching)

        self.word_dictionary.save(save_model)

        print("word count：", len(self.words))

    def eval(self, dataset, seg_lab=' ', w_file="test.txt", encoding="utf-8", threads=3):
        """
        评估。采用线程并非机制，读取测试数据集，并进行分词
        :param dataset: (str of list, mandatory) 测试数据.例如：['hello world']
        :param seg_lab: (str, optional, default=' ') 分词标记。在分词完成之后，需要根据某个字符标记词汇，默认为 " "
        :param w_file: (str, optional, default='test.txt') 分词结果写入的文件名
        :param encoding: (str, optional, default='utf-8') 分词结果写入文件的编码格式
        :param threads: (int, optional, default=3) 启动线程数量
        :return:
        """

        # 处理函数。主要使用正向和逆向匹配规则，然后进行消歧义。
        def handle(data, q):
            result = list()
            for sent in data:
                # forward_seg = self.matching.forward(d)
                # reverse_seg = self.matching.reverse(d)
                #
                # value = self.disambiguate.disambiguate([forward_seg, reverse_seg])
                # value = seg_lab.join(value)
                value = self.cut(sent, seg_lab)
                result.append(value)
                # 处理完一次结果，向消息队列写入 1
                q.put(1)
            return result

        # 预处理文本
        # texts = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)

        # 多线程处理
        multi_thread = MultiThreading(threads=threads)
        results = multi_thread.process(dataset, handle)

        # 将结果写入文件
        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, sent, seg_lab=' '):
        """
        输入某个句子或文本，对其进行分词
        :param sent: (str, mandatory) 句子文本
        :param seg_lab: (str, optional, default=' ') 分词完成之后需要通过某个字符对其进行标记，默认为 " "
        :return: (str) 分词文本
        """
        # text = DataTools.Preprocess.processing_text(data)
        forward_seg = self.matching.forward(sent)
        reverse_seg = self.matching.reverse(sent)

        result = self.disambiguate.disambiguate([forward_seg, reverse_seg])
        return seg_lab.join(result)

    @exception_handling
    def add_word(self, words, is_save=True, model_file='DictSegmentation.pickle'):
        """
        添加词汇到训练词典中
        :param words: (str or list or tuple, mandatory) 词汇，可以是字符、列表、元祖
        :param is_save: (bool, optional, default=True) 对于新加的词汇，是否保存词典
        :param model_file: (str, optional, default=DictSegmentation.pickle) 需要保存的模型文件名
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
        :param model_file: (str, mandatory) 保存的词典文件，也叫模型文件
        :param max_matching: (int, optional, default=10) 最大匹配词长度，用于正向和逆向匹配的最大匹配词长度
        :return: (DictSegmentation) 词典分词对象
        """
        dict_seg = DictSegmentation(max_matching)
        dict_seg.word_dictionary = WordDictionary.load(model_file)
        dict_seg.words = dict_seg.word_dictionary.get_dictionary()
        dict_seg.matching = Matching(dict_seg.words, max_matching)

        return dict_seg


def test_module_func():
    train_dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '的', '感情', '。'],
                     ['我', '扔', '了', '两颗', '手榴弹', '，', '他', '一下子', '出', '溜', '下去', '。'],
                     ['他', '要', '与', '中国人', '合作', '。']
                     ]
    test_dataset = ['他要与中国人合作。']
    model_file = "./dict_segmentation_model_debug.pickle"

    dict_seg = DictSegmentation(max_matching=10)
    dict_seg.fit(train_dataset, save_model=model_file)
    dict_seg.eval(test_dataset, seg_lab='  ', w_file='./test.txt', threads=1)

    sent = "他来到中国，成为第一个访华的大船主。"
    print("cut 1: ", dict_seg.cut(sent))

    dict_seg.add_word("中国", model_file=model_file)
    print("cut 2: ", dict_seg.cut(sent))

    load_dict_seg = DictSegmentation.load(model_file)
    assert load_dict_seg.cut(sent) == dict_seg.cut(sent), \
        "{} Error: load model segmentation error".format(DictSegmentation.load.__name__)


if __name__ == "__main__":
    test_module_func()
