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
        :param n: (int, optional, default=2) ngram 中的n元计算模式。n = {2,3}
        :param max_matching: (int, optional, default=2) 最大匹配长度默认=10
        """
        self.__max_matching = max_matching
        self.ngram_dict = NgramDictionary(n)
        self.ngram_disambiguate = None
        self.matching = None

    def fit(self, dataset, save_model="ngram_segmentation.pickle", smoothing=None, is_save=False):
        """
        拟合。输入训练数据集。创建词典、统计词频
        :param dataset: (list of list, mandatory) 文本数据
        :param save_model: (str, optional, default=ngram_segmentation.pickle) 保存模型
        :param smoothing: (str, optional, default=None) n 阶词频概率计算模式。目前只有 Laplace 模式
        :param is_save: (bool, optional, default=False) 是否保存训练的词典。这里自测保存词典容易出现内存错误。等待优化处理
        :return:
        """
        self.ngram_dict.fit(dataset)
        self.matching = Matching(self.ngram_dict.get_words(), max_matching=self.__max_matching)
        self.ngram_disambiguate = NgramDisambiguate(self.ngram_dict, smoothing)

        if is_save:
            self.ngram_dict.save(save_model)

    def eval(self, dataset, seg_lab="  ", w_file="test.txt", encoding="utf-8", threads=3):
        """
        评估。输入测试数据。对测试数据进行分词。
        :param dataset: (str, mandatory) 测试数据
        :param seg_lab: (str, optional, default=" ") 分词标记。默认"  "， 即在分词完成之后，使用该标记区分
        :param w_file: (str, optional, default="test.txt") 将结果写入的文件名
        :param encoding: (str, optional, default="utf-8") 写入文件的编码格式，默认为utf-8
        :param threads: (int, optional default=3) 线程数量
        :return:
        """

        def handle(data, q):
            result = list()
            for d in data:
                result.append(self.cut(d))
                q.put(1)
            return result

        # 多线程处理
        multi_thread = MultiThreading(threads=threads)
        results = multi_thread.process(dataset, handle)

        # 将结果写入文件
        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, sent, seg_lab=' ', del_start_str=None, del_end_str=None, regular_func=None):
        """
        分词。对某个文本或句子进行分词。
        :param sent: (str, mandatory) 文本
        :param seg_lab: (str, optional, default=" ") 分词标记。默认为 " " 。在分词完成之后，使用该标记标记完成的分词
        :param del_start_str: (str, optional, default=None) 需要删除的开始字符
        :param del_end_str: (str, optional, default=None) 需要删除的结束字符
        :param regular_func: (func, optional, default=None) 正则化函数
        :return: (str) 分词结果
        """
        forward_seg = self.matching.forward(sent)
        reverse_seg = self.matching.reverse(sent)

        forward_seg = [self.ngram_dict.START_TAG] + list(forward_seg) + [self.ngram_dict.END_TAG]
        reverse_seg = [self.ngram_dict.START_TAG] + list(reverse_seg) + [self.ngram_dict.END_TAG]

        best_text = self.ngram_disambiguate.disambiguate([forward_seg, reverse_seg])
        # 剔除 (START) 和 (END) 字符
        best_text = best_text[1:-1]

        return seg_lab.join(best_text)

    @staticmethod
    def load(model_file, n=NGRAM, smoothing=None, max_matching=MAX_MATCHING):
        """
        加载模型。使用保存的模型文件进行初始化。
        :param model_file: (str, mandatory) 模型文件
        :param n: (int, optional, default=2) ngram 中的n元计算模式，n = {2,3}
        :param smoothing: (str, optional, default=None) 概率计算模式，只支持 laplace
        :param max_matching: (int, optional, default=10) 最大匹配长度
        :return: (NgramSegmentation) 模型对象
        """
        ngram_seg = NgramSegmentation(n, max_matching=max_matching)
        ngram_seg.ngram_dict = NgramDictionary.load(model_file)
        ngram_seg.ngram_disambiguate = NgramDisambiguate(ngram_seg.ngram_dict, smoothing)
        ngram_seg.matching = Matching(ngram_seg.ngram_dict.get_words(), max_matching=max_matching)

        return ngram_seg


def test_module_func():
    train_dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '的', '感情', '。'],
                     ['我', '扔', '了', '两颗', '手榴弹', '，', '他', '一下子', '出', '溜', '下去', '。'],
                     ['他', '要', '与', '中国人', '合作', '。']
                     ]
    test_dataset = ['他要与中国人合作。']
    model_file = "./ngram_segmentation_model_debug.pickle"

    ngram_seg = NgramSegmentation(n=2)
    ngram_seg.fit(train_dataset, save_model=model_file, is_save=True)
    ngram_seg.eval(test_dataset, seg_lab="  ", w_file='./ngram_test_debug.utf8', threads=1)

    sent = "他要与中国人合作。"
    cut_sent = ngram_seg.cut(sent)
    print(ngram_seg.cut(sent))

    load_ngram_seg = NgramSegmentation.load(model_file)
    load_cut_sent = load_ngram_seg.cut(sent)
    assert cut_sent == load_cut_sent, "{} Error: load model segmentation error ".format(load_ngram_seg.cut.__name__)


if __name__ == "__main__":
    test_module_func()
