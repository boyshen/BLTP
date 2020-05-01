# -*- encoding: utf-8 -*-
"""
@file: writer.py
@time: 2020/4/21 下午4:42
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 1.写入文件。将相关的结果记录文件中
# 2.检查路径。检查写入的文件路径是否存在，不存在则创建
"""

import os

try:
    from .color import Color
    from .exception import exception_handling, ParameterError
except ModuleNotFoundError:
    from module.core.color import Color
    from module.core.exception import exception_handling, ParameterError


class Writer(object):
    # 系统目录分割符，linux 系统下分隔符为 "/"，windows 下目录分割符是 "\"
    Dir_Separator = "/"

    @staticmethod
    @exception_handling
    def writer_file(file, results, mode='a', encoding='utf-8'):
        """
        输入字符串、列表字符，写入文件
        :param file: <str> 文件名 或 路径 + 文件名
        :param results: <str, list, tuple> 需要写入文件的字符集或单个字符串。例如"hello word" 或['hello world']
        :param mode: <str> 模式。默认为 'a' ，追加模式
        :param encoding: <str> 编码。默认为 UTF-8 编码
        :return:
        """
        if not isinstance(results, (str, list, tuple)):
            raise ParameterError(
                "result parameter must be is {}, but actually get {}".format((str, list, tuple), type(results)))
        if isinstance(results, (list, tuple)):
            for result in results:
                if not isinstance(result, str):
                    raise ParameterError(
                        "results parameter elements must be is str, but actually get {}, elements:{}".format(
                            type(result), result))

        Writer.check_path(file)

        with open(file, mode, encoding=encoding) as f_write:
            if isinstance(results, str):
                f_write.writelines(results + '\n')
            else:
                for result in results:
                    f_write.writelines(result + '\n')

        print("\n" + "over！File: {}, encoding: {}".format(Color.red(file), Color.red(encoding)))

    @staticmethod
    def check_path(file):
        """
        检查文件目录是否存在，不存在则创建。
        :param file: <str> 文件所在路径。绝对路径。例如："／home/test／test.file"
        :return:
        """
        parameter_list = file.split(Writer.Dir_Separator)
        if len(parameter_list) != 1:
            path = file[:len(file) - len(parameter_list[-1])]
            if not os.path.exists(path):
                os.makedirs(path)


def test():
    Writer.writer_file("../../test/test.write.file", "hello word")
    Writer.writer_file("../../test/test/test.write.file", ["hello word", "hello world"])
    Writer.writer_file("/Users/shen/Desktop/me/python/AI/nlp/running/BLTP/test/test1/test.log", ("ha ha", "hello"))


if __name__ == "__main__":
    test()
