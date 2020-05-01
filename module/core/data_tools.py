# -*- encoding: utf-8 -*-
"""
@file: data_tools.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 数据预处理工具，主要负责对训练数据、测试数据、单个输入数据预处理。根据场景有分词、词性标注、命名实体识别
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
        @exception_handling
        def processing_text(text, del_start_str=None, del_end_str=None, regular_func=None):
            """
            预处理文本，用于处理输入句子。可删除句子的前后标记字符和对句子进行正则化处理。
            :param text: <str> 文本
            :param del_start_str: <str> 需要删除文本的开始标记字符
            :param del_end_str: <str> 需要删除文本的结束标记字符
            :param regular_func: <function> 回调正则化函数
            :return: <str> 预处理的文本
            """

            if del_start_str is not None and text[:len(del_start_str)] == del_start_str:
                text = text[len(del_start_str):]
                text = text.strip()

            if del_end_str is not None and text[-len(del_end_str):] == del_end_str:
                text = text[:-len(del_end_str)]
                text = text.strip()

            if regular_func is not None:
                text = regular_func(text)

            return text

        @staticmethod
        @exception_handling
        def read_file_data(file, del_start_str=None, del_end_str=None, regular_func=None):
            """
            读取文件里的数据。主要用于读取训练数据和测试数据，并对数据进行预处理。
            :param file: <str> 数据文件
            :param del_start_str: <str> 需要删除的开始标记字符
            :param del_end_str: <str> 需要删除的结束标记字符
            :param regular_func: <function> 回调正则化函数
            :return: <list> 预处理的数据列表
            """
            if not os.path.isfile(file):
                raise FileNotFoundException(file)

            data = list()
            with open(file, "rb") as lines:
                for line in tqdm(lines):
                    line = str(line, encoding="utf8").strip('\n').strip()

                    line = DataTools.Preprocess.processing_text(line, del_start_str, del_end_str, regular_func)

                    data.append(line)

            return data


def regular(sentence):
    """测试正则化函数"""
    # sentence = ''.join(re.findall('[\u4e00-\u9fa5,，.。？?！!·、{};；<>()<<>>《》0-9a-zA-Z\[\]]+', sentence))
    sentence = re.sub('[，]{1,100}', '，', sentence)
    sentence = re.sub('[,]{1,100}', ',', sentence)
    sentence = re.sub('[？]{1,100}', '？', sentence)
    sentence = re.sub('[?]{1,100}', '?', sentence)
    sentence = re.sub('[！]{1,100}', '！', sentence)
    sentence = re.sub('[!]{1,100}', '!', sentence)
    sentence = re.sub('[。]{1,100}', '。', sentence)
    sentence = re.sub('[.]{1,100}', '.', sentence)
    sentence = re.sub('[、]{1,100}', '、', sentence)
    sentence = re.sub('[：]{1,100}', '：', sentence)
    sentence = re.sub('[～]{1,100}', '', sentence)
    return sentence


def test_read_file_data():
    """测试函数"""
    file = "../../data/msr_training_debug.utf8"
    # file = "../../data/msr_test_debug.utf8"

    data = DataTools.Preprocess.read_file_data(file, del_start_str='“')
    # data = read_file_data(file, regular_func=regular)

    with open(file, "rb") as lines:
        for i, line in enumerate(lines):
            line = str(line, encoding="utf8").strip('\n')
            print("file:", line)
            print("read data:", data[i])
            print()


if __name__ == "__main__":
    test_read_file_data()
