# -*- encoding: utf-8 -*-
"""
@file: segmentation.py
@time: 2020/4/21 上午11:11
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:  分词类。定义 fit、eval、cut、load方法。将通过子类实现
"""


class Segmentation(object):
    def fit(self, dataset, save_model="segmentation.pickle"):
        pass

    def eval(self, dataset, seg_lab=" ", w_file="test.txt", encoding="utf-8"):
        pass

    def cut(self, text, seg_lab=" "):
        pass

    @staticmethod
    def load(model_file):
        pass
