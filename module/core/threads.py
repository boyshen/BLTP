# -*- encoding: utf-8 -*-
"""
@file: threads.py
@time: 2020/4/16 上午9:37
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 自定义线程模块
# 多线程处理任务
"""

import threading
import sys
from queue import Queue
from time import sleep

try:
    from .data_tools import DataTools
except ModuleNotFoundError:
    from module.core.data_tools import DataTools


class ThreadFunc(threading.Thread):

    def __init__(self, thread_name, thread_id, task_total, func, *args):
        super(ThreadFunc, self).__init__()
        self.thread_name = thread_name
        self.thread_id = thread_id
        self.__func = func
        self.__args = args

        self.task_total = task_total
        self.process = 0

        self.__result = None

    def run(self):
        print("{}_{} start...".format(self.thread_name, self.thread_id))
        result = self.__func(*self.__args)
        self.__result = result

    def get_result(self):
        return self.__result


class MultiThreading(object):
    SLEEP_SEC = 0.5
    QUEUE_MAX_SIZE = 10240

    def __init__(self, threads):
        # 定义线程数量
        self.threads = threads

    def process(self, data, handle_func):
        """
        多线程处理数据集
        :param data: <list, tuple> 数据集
        :param handle_func: <func> 定义线程处理函数。
        # 函数格式要求有两个参数，一个是处理数据集，一个队列。例如：
        # def handle(data, q):
        #   for d in data:
        #       # 处理数据 d
        #       # q 是队列，这里处理完一次，put(1) 用于显示处理进度
        #       q.put(1)
        :return: <list> 处理结果
        """
        # 预处理文本
        data_num = len(data)

        # 计算每个线程平均处理的样本
        sample_num = int(data_num / self.threads)
        threads_pool = []
        queue_pool = []

        # 初始化线程任务
        for i in range(self.threads):
            if i + 1 == self.threads:
                sample = data[sample_num * i:]
            else:
                sample = data[sample_num * i: sample_num * (i + 1)]

            # 初始化任务队列，用于显示处理进度.
            task_queue = Queue(MultiThreading.QUEUE_MAX_SIZE)

            thread = ThreadFunc("thread", i, len(sample), handle_func, sample, task_queue)
            threads_pool.append(thread)
            queue_pool.append(task_queue)

        # 开启线程
        for thread in threads_pool:
            thread.start()

        # 显示处理进度。通过线程队列机制。将每个线程的处理进度通过队列put，在这里进行get。并进行输出显示
        total_process = 0
        while True:
            # 动态输出结果，通常情况下每隔0.5秒更新一次结果
            sleep(MultiThreading.SLEEP_SEC)
            show = ""
            for thread, t_queue in zip(threads_pool, queue_pool):
                if not t_queue.empty():
                    thread.process += t_queue.get(timeout=5)
                    t_queue.task_done()
                    total_process += 1
                show = show + "{}_{} Process:{}/{} \t".format(thread.thread_name, thread.thread_id,
                                                              thread.process,
                                                              thread.task_total)
            percentage = (total_process / data_num) * 100
            sys.stdout.write("\r" + show + "Total process: {}/{} Percentage:{:.2f}%".format(total_process, data_num,
                                                                                            percentage))
            sys.stdout.flush()

            if total_process == data_num:
                break
        print()

        # 等待线程执行完成
        for thread in threads_pool:
            thread.join()

        # 等待队列处理完成
        for t_queue in queue_pool:
            t_queue.join()

        # 获取处理结果
        handle_result = list()
        for thread in threads_pool:
            handle_result = handle_result + thread.get_result()

        return handle_result
