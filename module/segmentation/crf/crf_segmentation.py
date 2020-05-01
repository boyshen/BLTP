# -*- encoding: utf-8 -*-
"""
@file: crf_segmentation.py
@time: 2020/4/28 下午3:58
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 1. extract_feature 提取特征。输入句子或文本，将句子文本转换成特征值。
# 2. encoding 编码。将使用 " " 或 "/" 分词的文本转换成 "BMES" 格式。
# 3. decoding 解码。将使用 "BMES" 格式标记的词转换成字符列表。
# 4. fit 拟合模型。
# 5. eval 评估。
# 6. cut 分词。
# 8. save 保存模型。
# 9. load 加载模型。
参考 sklearn_crfsuite 文档：https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
"""
import re
import sklearn_crfsuite
import pickle
import os
from tqdm import tqdm
from module.core.segmentation import Segmentation
from module.core.data_tools import DataTools
from module.core.exception import exception_handling, ParameterError, FileNotFoundException, UnknownError
from module.core.writer import Writer
from module.core.threads import MultiThreading


class CRFSegmentation(Segmentation):
    B_TAG = "B"
    M_TAG = "M"
    E_TAG = "E"
    S_TAG = "S"

    def __init__(self, algorithm='lbfgs', min_freq=0, c1=0, c2=1.0, max_iterations=None):
        """
        选用 sklearn_crfsuite AIP 文档中常用参数。详细的参数信息，可以参考 sklearn_crfsuite API 文档。
        sklearn_crfsuite API 文档：https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
        """
        super(CRFSegmentation, self).__init__()
        self.__crf = sklearn_crfsuite.CRF(algorithm=algorithm,
                                          min_freq=min_freq,
                                          c1=c1,
                                          c2=c2,
                                          max_iterations=max_iterations)

    @staticmethod
    def extract_feature(sent):
        """
        提取特征.
        参考 sklearn_crfsuite 文档：https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
        将输入的句子转换成列表字典格式。其中字典的 key 如下：
            'word'： 为当前词，
            '-1:word'： 为上一个词。
            '+1:word'： 为下一个词。
            'BOS'：代表该词是句子开头的第一个词。
            'EOS'  代表该词是句子的最后一个词。
        例如：
            输入：
            输出：
        :param sent: (str, mandatory) 句子或文本
        :return: (list of dict) 列表字典。
        """
        sent = re.sub('[ ]+', '', sent)
        sent = list(sent)
        features = list()
        for i, word in enumerate(sent):
            feature = {"bias": 1.0, "word": word}
            if i == 0:
                feature['BOS'] = True
            else:
                feature["-1:word"] = sent[i - 1]

            if i + 1 == len(sent):
                feature["EOS"] = True
            else:
                feature["+1:word"] = sent[i + 1]
            features.append(feature)
        return features

    @staticmethod
    def encoding(sent, split_lab=" "):
        """
        编码。将采用" " 或 "／" 分词文本转换成 "BMES" 格式。
        例如：
            输入："小明 是 中国人"
            输出：['B','E','S','B','M','E']
        :param sent: (str, mandatory)
        :param split_lab: (str, optional，default="  ") 分词标签。
        :return: (list of str) 列表字符
        """
        sent = sent.strip().split(split_lab)
        label = list()
        for word in sent:
            if word == "" or word == " " or len(word) == 0:
                continue
            word = word.strip()
            if len(word) == 1:
                label += [CRFSegmentation.S_TAG]
            elif len(word) == 2:
                label += [CRFSegmentation.B_TAG, CRFSegmentation.E_TAG]
            elif len(word) > 2:
                label += [CRFSegmentation.B_TAG] + [CRFSegmentation.M_TAG] * len(word[1:-1]) + [CRFSegmentation.E_TAG]
        return label

    @staticmethod
    @exception_handling
    def decoding(sent, label):
        """
        解码。将采用 "BMES" 分词的文本或句子转换成字符列表
        例如：
            输入：['小','明','是','中','国','人'] 和 ['B','E','S','B','M','E']
            输出：["小明","是","中国人"]
        :param sent: (list, mandatory) 字符列表
        :param label: (list, mandatory) 标签列表
        :return: (list) 字符列表
        """
        if len(sent) != len(label):
            raise ParameterError("sent length:{} and label length:{} must be equal".format(len(sent), len(label)))

        words = []
        string = ""
        for word, lab in zip(sent, label):
            if lab == CRFSegmentation.S_TAG:
                words.append(word)
            elif lab == CRFSegmentation.B_TAG and string == "":
                string += word
            elif lab == CRFSegmentation.M_TAG and string != "":
                string += word
            elif lab == CRFSegmentation.E_TAG and string != "":
                string += word
                words.append(string)
                string = ""
        return words

    @exception_handling
    def fit(self, file, split_lab, save_model="crf_segmentation.pickle", del_start_str=None, del_end_str=None,
            regular_func=None):
        """
        拟合模型
        :param file: (str, mandatory) 训练数据
        :param split_lab: (str, mandatory) 训练文本中对词的划分标记
        :param save_model: (str, optional, default='crf_segmentation.pickle') 保存模型的文件名
        :param del_start_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: (fun, optional, default=None)> 正则化函数
        :return:
        """

        text = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)
        train_x, train_y = list(), list()
        for sent in tqdm(text):
            feature = CRFSegmentation.extract_feature("".join(sent.split(split_lab)))
            label = CRFSegmentation.encoding(sent, split_lab=split_lab)
            if len(feature) != len(label):
                raise UnknownError(
                    "Unknown error , Please check code! feature length: {}, label length: {}, sent: {}".format(
                        len(feature), len(label), sent))
            train_x.append(feature)
            train_y.append(label)

        self.__crf.fit(train_x, train_y)

        self.save(save_model)

    def eval(self, file, seg_lab="  ", w_file="test.txt", encoding="utf-8", threads=3, del_start_str=None,
             del_end_str=None, regular_func=None):
        """
        评估。
        :param file: (str, mandatory) 测试数据
        :param seg_lab: (str, mandatory, default="  ") 分词完成之后使用该标记区分
        :param w_file: (str, optional, default="test.txt") 将分词结果写入文件
        :param encoding: (str, optional, default="utf-8") 写入文件的编码格式
        :param threads: (int optional, default=3) 执行线程数量
        :param del_start_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: (fun, optional, default=None)> 正则化函数
        :return:
        """

        def handle(data, q):
            result = list()
            for sent in data:
                feature = CRFSegmentation.extract_feature(sent)
                predict_label = self.__crf.predict_single(feature)
                words = CRFSegmentation.decoding(sent, predict_label)
                value = seg_lab.join(words)
                result.append(value)

                q.put(1)
            return result

        text = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, regular_func)

        multi_threads = MultiThreading(threads=threads)
        results = multi_threads.process(text, handle)

        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, text, seg_lab=' ', del_start_str=None, del_end_str=None, regular_func=None):
        """
        分词。
        :param text: (str, mandatory) 字符文本或句子
        :param seg_lab: (str, optional, default="  ") 分词完成之后使用该标记区分
        :param del_start_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在开始标记需要删除，如果有，则输入
        :param del_end_str: (str, optional, default=None) 对于训练数据中的文本句子，是否存在结束标记需要删除，如果有，则输入
        :param regular_func: (fun, optional, default=None)> 正则化函数
        :return: (str) 分词句子
        """
        text = DataTools.Preprocess.processing_text(text, del_start_str, del_end_str, regular_func)
        feature = CRFSegmentation.extract_feature(text)
        predict_label = self.__crf.predict_single(feature)
        words = CRFSegmentation.decoding(text, predict_label)

        return seg_lab.join(words)

    @staticmethod
    def load(file):
        """
        加载模型。
        :param file: (str, mandatory) 模型文件
        :return: (MFSSegmentation) 分词对象
        """
        if not os.path.isfile(file):
            raise FileNotFoundException(file)

        with open(file, "rb") as f_read:
            model = pickle.loads(f_read.read())

        return model

    def save(self, save_model="crf_segmentation.pickle"):
        """
        保存模型。
        :param save_model:(str, optional, default="crf_segmentation.pickle")
        :return:
        """
        file = save_model.strip()

        Writer.check_path(file)

        pickle.dump(self, open(file, "wb"))

        print("Save model over! File: {}".format(file))


def test():
    train_file = "../../../data/msr_training_debug.utf8"
    # train_file = "../../../data/icwb2-data/training/msr_training.utf8"
    save_model = "../../../model/crf_segmentation_debug.pickle"
    crf = CRFSegmentation()
    crf.fit(train_file, split_lab=" ", save_model=save_model, del_start_str="“")

    w_file = "../../../result/crf_test_debug.utf8"
    test_file = "../../../data/msr_test_debug.utf8"
    crf.eval(test_file, seg_lab="  ", w_file=w_file)


def test_load():
    save_model = "../../../model/crf_segmentation_debug.pickle"
    crf = CRFSegmentation.load(save_model)

    sent = "种田要有个明白账，投本要赚利润是起码的道理。"
    result = crf.cut(sent, seg_lab="/")
    print("sent: ", sent)
    print("predict: ", result)


if __name__ == "__main__":
    test()
    # test_load()
