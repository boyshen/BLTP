# -*- encoding: utf-8 -*-
"""
@file: data_tools.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:数据预处理工具，主要负责对训练数据、测试数据、单个输入数据预处理。根据场景有分词、词性标注、命名实体识别
# 1. Preprocess.processing_text 预处理输入的句子或文本。
# 2. Preprocess.read_file_data 读取普通文件数据。
# 3. Preprocess.read_json_data 读取json文件数据。
"""

import os
import re
from tqdm import tqdm

try:
    from module.core.exception import exception_handling, \
        FileNotFoundException
except ModuleNotFoundError:
    from .exception import exception_handling, \
        FileNotFoundException


class DataTools(object):
    """
    数据处理工具集
    """

    def __init__(self):
        pass

    class Preprocess(object):
        """
        分词预处理类
        """

        def __init__(self):
            pass

        @staticmethod
        def processing_text(text, del_start_str=None, del_end_str=None, handle_func=None):
            """
            预处理文本，用于处理输入句子。可删除句子的前后标记字符和对句子进行正则化处理。
            :param text: (str, mandatory) 文本
            :param del_start_str: (list or str, optional, default=None) 需要删除文本的开始标记字符
            :param del_end_str: (list or str, optional, default=None) 需要删除文本的结束标记字符
            :param handle_func: (function, optional, default=None) 回调处理函数，对读取的每行数据进行处理。
            :return: (str) 预处理的文本
            """
            text = text.strip('\n').strip()
            if del_start_str is not None:
                if isinstance(del_start_str, (list, tuple)):
                    for start_str in del_start_str:
                        if text[:len(start_str)] in del_start_str:
                            text = text[len(start_str):]
                            text = text.strip()

                elif isinstance(del_start_str, str):
                    if text[:len(del_start_str)] == del_start_str:
                        text = text[len(del_start_str):]
                        text = text.strip()

            if del_end_str is not None:
                if isinstance(del_end_str, (list, tuple)):
                    for end_str in del_end_str:
                        if text[-len(end_str):] in del_end_str:
                            text = text[:-len(end_str)]
                            text = text.strip()

                elif isinstance(del_end_str, str):
                    if text[-len(del_end_str):] == del_end_str:
                        text = text[:-len(del_end_str)]
                        text = text.strip()

            if handle_func is not None:
                text = handle_func(text)

            return text

        @staticmethod
        @exception_handling
        def read_file_data(file, del_start_str=None, del_end_str=None, handle_func=None):
            """
            读取文件里的数据。主要用于读取训练数据和测试数据。通过行读取的方式读取数据集。
            :param file: (str, mandatory) 数据文件
            :param del_start_str: (str, optional, default=None) 需要删除的开始标记字符
            :param del_end_str: (str, optional, default=None) 需要删除的结束标记字符
            :param handle_func: (function, optional, default=None) 回调处理函数，对读取的每行数据进行处理。
            :return: (list) 预处理的数据列表
            """
            if not os.path.isfile(file):
                raise FileNotFoundException(file)

            data = list()
            with open(file, "rb") as lines:
                for line in tqdm(lines):
                    line = str(line, encoding="utf8")

                    line = DataTools.Preprocess.processing_text(line, del_start_str, del_end_str, handle_func)

                    data.append(line)

            return data


def test_read_file_data(file, del_start_str=None, del_end_str=None, handle_func=None):
    """测试函数"""
    data = DataTools.Preprocess.read_file_data(file, del_start_str, del_end_str, handle_func)
    with open(file, "rb") as lines:
        for source_data, target_data in zip(lines, data):
            source_data = str(source_data, encoding="utf8").strip('\n')
            print("source data:", source_data)
            print("target data:", target_data)
            print()


if __name__ == "__main__":
    def regular(sent):
        """测试正则化函数"""
        sent = re.sub('[，]{1,100}', '，', sent)
        return sent


    def handle(line):
        """测试数据处理函数"""
        import json

        json_data = json.loads(line)
        text = json_data['text']
        label = json_data['label']

        identifier_b, identifier_i, identifier_o, identifier_e, identifier_s = "B", "I", "O", "E", "S"
        identifier_format = lambda i, s: "{}_{}".format(i, s)
        identifier = [identifier_o] * len(text)

        for ner_name, ner_value in label.items():
            for ner_str, ner_index in ner_value.items():
                for n_index in ner_index:
                    if text[n_index[0]:n_index[1] + 1] != ner_str:
                        print("Data Error: no specific character found . text: {}, label: {}".format(text, label))
                        exit()
                    if len(ner_str) == 1:
                        identifier[n_index[0]] = identifier_format(identifier_o, ner_name)
                    elif len(ner_str) == 2:
                        identifier[n_index[0]] = identifier_format(identifier_b, ner_name)
                        identifier[n_index[1]] = identifier_format(identifier_e, ner_name)
                    elif len(ner_str) > 2:
                        identifier[n_index[0]] = identifier_format(identifier_b, ner_name)
                        for i in range(1, len(ner_str) - 2 + 1):
                            identifier[n_index[0] + i] = identifier_format(identifier_i, ner_name)
                        identifier[n_index[1]] = identifier_format(identifier_e, ner_name)

        return [text, identifier]


    # train_file = "../../data/msr_training_debug.utf8"
    # test_read_file_data(train_file, del_start_str=["“", "’"], handle_func=regular)
    dev_file = "../../data/cluener_public/dev.json"
    test_read_file_data(dev_file, handle_func=handle)
