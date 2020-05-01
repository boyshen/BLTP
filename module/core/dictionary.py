# -*- encoding: utf-8 -*-
"""
@file: dictionary.py
@time: 2020/4/7 下午5:57
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 词典的父类对象，主要提供给子类进行继承，提供常用的序列标记 <START> 和 <END> 以及保存和加载词典。
"""

import os
import re
import pickle

try:
    from module.core.exception import exception_handling, \
        FileNotFoundException
    from module.core.writer import Writer
except ModuleNotFoundError:
    from .exception import exception_handling, \
        FileNotFoundException
    from .writer import Writer


class Dictionary(object):
    """
    词典对象的父类，提供对词典的保存和加载函数，以及对标点符号的处理函数。
    """
    # 序列开始标记、结束标记、未知标记
    START_TAG = "<START>"
    END_TAG = "<END>"
    UNK_TAG = "<UNK>"

    # 序列标记对应的值
    UNK = 0
    START = 1
    END = 2

    def __init__(self):
        pass

    def fit(self, texts, split_lab):
        pass

    @staticmethod
    def delete_punctuation(text):
        """
        使用正则表达式删除文本中的标点符号
        :param text: <str> 字符串格式的文本
        :return: <str> 处理后的文本
        """
        regular = "[，、“”《》（）：？！；。…—]"
        text = re.sub(regular, '', text)
        return text

    def save(self, file):
        """
        保存拟合的词典
        :param file: <str> 保存路径+文件名
        :return: 无
        """
        # 去除文件路径的开头和结尾的空白字符
        file = file.strip()

        # 检查目录是否存在，不存在则创建
        Writer.check_path(file)

        # 保存
        pickle.dump(self, open(file, "wb"))

        print("save dictionary success! File: {}".format(file))

    @staticmethod
    @exception_handling
    def load(file):
        """
        通过保存的词典文件加载词典对象
        :param file: <str> 词典文件
        :return: <Dictionary> 词典对象
        """
        if not os.path.isfile(file):
            raise FileNotFoundException(file)

        with open(file, "rb") as f_read:
            dictionary = pickle.loads(f_read.read())

        return dictionary
