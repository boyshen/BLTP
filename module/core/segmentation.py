# -*- encoding: utf-8 -*-
"""
@file: segmentation.py
@time: 2020/4/21 上午11:11
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:  分词类。定义 fit、eval、cut、load方法。将通过子类实现
"""


class Segmentation(object):
    def fit(self, file, split_lab, save_model="segmentation.pickle", del_start_str=None, del_end_str=None,
            regular_func=None):
        pass

    def eval(self, file, seg_lab="  ", w_file="test.txt", encoding="utf-8", del_start_str=None,
             del_end_str=None, regular_func=None):
        pass

    def cut(self, text, seg_lab=' ', del_start_str=None, del_end_str=None, regular_func=None):
        pass

    @staticmethod
    def load(model_file):
        pass
