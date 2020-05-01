# -*- encoding: utf-8 -*-
"""
@file: ngram_segmentation.py
@time: 2020/4/20 下午9:29
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: ngram 分词模型
# 1. 拟合。输入训练数据集，创建词典、统计词频。
# 2. 评估。输入测试数据集，将分词结果写入文本中
# 3. 分词。输入单个文本或语句。对文本或语句进行分词
# 4. 加载。加载模型
"""

from module.segmentation.ngram.ngram_dictionary import NgramDictionary
from module.segmentation.ngram.ngram_disambiguate import NgramDisambiguate
from module.core.data_tools import DataTools
from module.core.matching import Matching
from module.core.threads import MultiThreading
from module.core.segmentation import Segmentation
from module.core.writer import Writer

NGRAM = 2
MAX_MATCHING = 10


class NgramSegmentation(Segmentation):

    def __init__(self, n=NGRAM, max_matching=MAX_MATCHING):
        """
        初始化
        :param n: <int> ngram 中的n元计算模式。n = {2,3}
        :param max_matching: <int> 最大匹配长度默认=10
        """
        self.__max_matching = max_matching
        self.ngram_dict = NgramDictionary(n)
        self.ngram_disambiguate = None
        self.matching = None

    def fit(self, file, split_lab, save_model="ngram_segmentation.pickle", smoothing=None, del_start_str=None,
            del_end_str=None, regular_func=None, is_save=False):
        """
        拟合。输入训练数据集。创建词典、统计词频
        :param file: <str> 文本数据
        :param split_lab: <str> 训练数据集中的分词标记
        :param save_model: <str> 保存模型
        :param smoothing: <str> n 阶词频概率计算模式。目前只有 Laplace 模式
        :param del_start_str: <str> 需要删除的开始字符
        :param del_end_str: <str> 需要删除的结束字符
        :param regular_func: <func> 正则化函数
        :param is_save: <bool> 是否保存训练的词典。这里自测保存词典容易出现内存错误。等待优化处理
        :return:
        """
        texts = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)
        self.ngram_dict.fit(texts, split_lab)
        self.matching = Matching(self.ngram_dict.get_words(), max_matching=self.__max_matching)
        self.ngram_disambiguate = NgramDisambiguate(self.ngram_dict, smoothing)

        if is_save:
            self.ngram_dict.save(save_model)

    def eval(self, file, seg_lab="  ", w_file="test.txt", encoding="utf-8", threads=3, del_start_str=None,
             del_end_str=None, regular_func=None):
        """
        评估。输入测试数据。对测试数据进行分词。
        :param file: <str> 测试数据文件名
        :param seg_lab: <str> 分词标记。默认"  "， 即在分词完成之后，使用该标记区分
        :param w_file: <str> 将结果写入的文件名
        :param encoding: <str> 写入文件的编码格式，默认为utf-8
        :param threads: <int> 线程数量
        :param del_start_str: <str> 需要删除的开始字符
        :param del_end_str: <str> 需要删除的结束字符
        :param regular_func: <func> 正则化函数
        :return:
        """

        def handle(data, q):
            result = list()
            for d in data:
                forward_seg = self.matching.forward(d)
                reverse_seg = self.matching.reverse(d)

                forward_seg = [self.ngram_dict.START_TAG] + list(forward_seg) + [self.ngram_dict.END_TAG]
                reverse_seg = [self.ngram_dict.START_TAG] + list(reverse_seg) + [self.ngram_dict.END_TAG]

                best_text = self.ngram_disambiguate.disambiguate([forward_seg, reverse_seg])
                # 剔除 <START> 和 <END> 字符
                best_text = best_text[1:-1]
                result.append(seg_lab.join(best_text))
                q.put(1)

            return result

        texts = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)

        # 多线程处理
        multi_thread = MultiThreading(threads=threads)
        results = multi_thread.process(texts, handle)

        # 将结果写入文件
        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, text, seg_lab=' ', del_start_str=None, del_end_str=None, regular_func=None):
        """
        分词。对某个文本或句子进行分词。
        :param text: <str> 文本
        :param seg_lab: <str> 分词标记。默认为 " " 。在分词完成之后，使用该标记标记完成的分词
        :param del_start_str: <str> 需要删除的开始字符
        :param del_end_str: <str> 需要删除的结束字符
        :param regular_func: <func> 正则化函数
        :return: <str> 分词结果
        """
        text = DataTools.Preprocess.processing_text(text, del_start_str, del_end_str, regular_func)
        forward_seg = self.matching.forward(text)
        reverse_seg = self.matching.reverse(text)

        forward_seg = [self.ngram_dict.START_TAG] + list(forward_seg) + [self.ngram_dict.END_TAG]
        reverse_seg = [self.ngram_dict.START_TAG] + list(reverse_seg) + [self.ngram_dict.END_TAG]

        best_text = self.ngram_disambiguate.disambiguate([forward_seg, reverse_seg])
        # 剔除 <START> 和 <END> 字符
        best_text = best_text[1:-1]

        return seg_lab.join(best_text)

    @staticmethod
    def load(model_file, n=NGRAM, smoothing=None, max_matching=MAX_MATCHING):
        """
        加载模型。使用保存的模型文件进行初始化。
        :param model_file: <str> 模型文件
        :param n: <int> ngram 中的n元计算模式，n = {2,3}
        :param smoothing: <str> 概率计算模式，只支持 laplace
        :param max_matching: <int> 最大匹配长度
        :return: <NgramSegmentation> 模型对象
        """
        ngram_seg = NgramSegmentation(n, max_matching=max_matching)
        ngram_seg.ngram_dict = NgramDictionary.load(model_file)
        ngram_seg.ngram_disambiguate = NgramDisambiguate(ngram_seg.ngram_dict, smoothing)
        ngram_seg.matching = Matching(ngram_seg.ngram_dict.get_words(), max_matching=max_matching)

        return ngram_seg


def test():
    # train_file = "../../../data/msr_training_debug.utf8"
    train_file = "../../../data/icwb2-data/training/msr_training.utf8"
    test_file = "../../../data/msr_test_debug.utf8"

    save_model = "../../../model/ngram_segmentation_debug.pickle"
    w_file = "../../../result/ngram_test_debug.utf8"

    ngram_seg = NgramSegmentation(n=2)
    ngram_seg.fit(train_file, split_lab='  ', save_model=save_model, del_start_str="“")
    ngram_seg.eval(test_file, seg_lab="  ", w_file=w_file)


def test_load():
    save_model = "../../../model/ngram_segmentation_debug.pickle"
    ngram_seg = NgramSegmentation.load(model_file=save_model, n=2)
    print("words number: ", len(ngram_seg.ngram_dict.get_words()))

    text = "从这一点说，我理解你们的埋怨情绪。"
    seg_text = ngram_seg.cut(text)
    print("text:", text)
    print("seg text:", seg_text)


if __name__ == "__main__":
    test()
    # test_load()
