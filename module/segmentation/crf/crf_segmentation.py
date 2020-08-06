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
    def encoding(sent):
        """
        编码。转换成 "BMES" 分词格式
        例如：
            输入："小明, 是, 中国人"
            输出：['B','E','S','B','M','E']
        :param sent: (str, mandatory)
        :return: (list of str) 列表字符
        """
        # sent = sent.strip().split(split_lab)
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
    def fit(self, dataset, save_model="crf_segmentation.pickle"):
        """
        拟合模型
        :param dataset: (list of list, mandatory) 训练数据
        :param save_model: (str, optional, default='crf_segmentation.pickle') 保存模型的文件名
        :return:
        """
        train_x, train_y = list(), list()
        for sent in tqdm(dataset):
            feature = CRFSegmentation.extract_feature("".join(sent))
            label = CRFSegmentation.encoding(sent)
            if len(feature) != len(label):
                raise UnknownError(
                    "Please check code! feature length: {}, label length: {}, sent: {}".format(
                        len(feature), len(label), sent))
            train_x.append(feature)
            train_y.append(label)

        self.__crf.fit(train_x, train_y)

        self.save(save_model)

    def eval(self, dataset, seg_lab="  ", w_file="test.txt", encoding="utf-8", threads=3):
        """
        评估。
        :param dataset: (list of list, mandatory) 测试数据
        :param seg_lab: (str, mandatory, default="  ") 分词完成之后使用该标记区分
        :param w_file: (str, optional, default="test.txt") 将分词结果写入文件
        :param encoding: (str, optional, default="utf-8") 写入文件的编码格式
        :param threads: (int optional, default=3) 执行线程数量
        :return:
        """

        def handle(data, q):
            result = list()
            for sent in data:
                # feature = CRFSegmentation.extract_feature(sent)
                # predict_label = self.__crf.predict_single(feature)
                # words = CRFSegmentation.decoding(sent, predict_label)
                # value = seg_lab.join(words)
                value = self.cut(sent, seg_lab)
                result.append(value)

                q.put(1)
            return result

        multi_threads = MultiThreading(threads=threads)
        results = multi_threads.process(dataset, handle)

        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, sent, seg_lab=' '):
        """
        分词。
        :param sent: (str, mandatory) 字符文本或句子
        :param seg_lab: (str, optional, default="  ") 分词完成之后使用该标记区分
        :return: (str) 分词句子
        """
        sent = re.sub('[ ]+', '', sent)
        feature = CRFSegmentation.extract_feature(sent)
        predict_label = self.__crf.predict_single(feature)
        words = CRFSegmentation.decoding(sent, predict_label)

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


def test_module_func():
    train_dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '的', '感情', '。'],
                     ['我', '扔', '了', '两颗', '手榴弹', '，', '他', '一下子', '出', '溜', '下去', '。'],
                     ['他', '要', '与', '中国人', '合作', '。']
                     ]
    test_dataset = ['他要与中国人合作。']
    model_file = "./crf_segmentation_model_debug.pickle"

    crf_seg = CRFSegmentation()
    crf_seg.fit(train_dataset, save_model=model_file)
    crf_seg.eval(test_dataset, seg_lab='  ', w_file='./test.txt', threads=1)

    sent = "他来到中国，成为第一个访华的大船主。"
    print("cut : ", crf_seg.cut(sent))

    load_dict_seg = CRFSegmentation.load(model_file)
    assert load_dict_seg.cut(sent) == crf_seg.cut(sent), \
        "{} Error: load model segmentation error".format(CRFSegmentation.load.__name__)


if __name__ == "__main__":
    test_module_func()
