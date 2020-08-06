# -*- encoding: utf-8 -*-
"""
@file: hmm_segmentation.py
@time: 2020/4/23 下午5:52
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
1. fit 拟合模型。输入训练数据集。创建发射概率矩阵和转移概率矩阵
2. eval 评估。输入测试数据集。对测试数据进行分词，将分词结果写入文本
3. cut 分词。输入文本，对文本进行分词
4. load 加载模型。加载训练好的模型
"""

from module.core.segmentation import Segmentation
from module.core.exception import exception_handling, ParameterError, UnknownError
from module.segmentation.hmm.hmm_dictionary import HmmDictionary
from module.core.threads import MultiThreading
from module.core.writer import Writer


class Markov(object):
    """马尔可夫链。保存马尔可夫链每个状态的值"""

    class StateNode(object):
        def __init__(self, state=None, word=None, launch_prob=None, transition_prob=None, prob_product=None):
            # 保存当前状态
            self.state = state
            # 保存当前状态的字
            self.word = word
            # 保存当前状态的发射概率
            self.launch_prob = launch_prob
            # 保存上一个状态节点到当前状态节点的转移概率。格式：{'小_0_H': 0.2}，小_0_H 为上一个节点的mid值。
            # 小_0_H 中 小 为上一个字，0 为上一个字id，H 当前字的状态
            self.transition_prob = transition_prob
            # 保存之前节点到当前节点的概率积。如：当前节点为 E 。保存 (START) -> B -> M -> E 的发射概率与转移概率的乘积。必须是唯一路径
            self.prob_product = prob_product

    def __init__(self):
        self.markov_link = dict()

        self.mid_format = lambda w, w_i, s_i: "{}_{}_{}".format(w, w_i, s_i)

    @exception_handling
    def append(self, mid, state, word=None, launch_prob=None, transition_prob=None, prob_product=None):
        """添加节点"""
        if mid in self.markov_link.keys():
            raise ParameterError("mid={} already exists !".format(mid))

        self.markov_link[mid] = Markov.StateNode(state, word, launch_prob, transition_prob, prob_product)

    @exception_handling
    def delete(self, mid):
        """删除节点"""
        if mid not in self.markov_link.keys():
            raise ParameterError("mid={} not found !".format(mid))

        self.markov_link.pop(mid)

    def mid_is_exists(self, mid):
        """检查mid是否存在"""
        if mid in self.markov_link.keys():
            return True
        return False

    def get(self, mid):
        """获取节点对象"""
        if mid not in self.markov_link.keys():
            raise ParameterError("mid={} not found !".format(mid))

        return self.markov_link[mid]


class HmmSegmentation(Segmentation):

    def __init__(self):
        """
        初始化
        """
        super(HmmSegmentation, self).__init__()
        self.hmm_dict = HmmDictionary()

    def fit(self, dataset, save_model="hmm_segmentation.pickle"):
        """
        拟合
        :param dataset: (list of list, mandatory) 训练文件数据
        :param save_model: (str, optional, default=hmm_segmentation.pickle) 训练完成之后保存模型的文件名
        :return:
        """
        self.hmm_dict.fit(dataset)

        self.hmm_dict.save(save_model)

    @exception_handling
    def __viterbi(self, text):
        """
        维特比算法。输入文本列表。例如：['小', '明', '是', '中', '国', '人'], 返回 ['B', 'E', 'S', 'B', 'M', 'E']
        :param text: (list, mandatory) 文本列表
        :return: (list) 编码标签
        """
        # 为文本加上结束字符
        text = text + [self.hmm_dict.END_TAG]

        # 初始化马尔可夫链
        markov = Markov()
        # 初始化开始状态
        mid = self.hmm_dict.START_TAG
        markov.append(mid=mid, state=self.hmm_dict.START_TAG)

        save_last_state_mid = [mid]
        for i, word in enumerate(text):
            # 判断是否为最后一个字 (End) .如果是则为True，否则为 False
            is_end_word = True if word == self.hmm_dict.END_TAG else False

            # 保存该轮次最后状态的mid
            save_state_mid = []

            # 上一轮次最后保存的状态mid值
            for last_state_mid in save_last_state_mid:

                # 根据上一个节点的状态，获取到下一个节点状态和转移概率
                last_state = markov.get(last_state_mid).state
                state_transition_prob = self.hmm_dict.get_state_transition_prob(last_state, is_end_word)

                # 如果获取到的发射概率为空，则进行下一轮。这种情况将可能出现在最后一个字，最后一个字到(End)
                if len(state_transition_prob) == 0:
                    continue

                for state, prob in state_transition_prob.items():
                    mid = markov.mid_format(word, i, state)
                    launch_prob = None
                    # 如果下一个节点已经存在，则说明有两条路径到该节点，则采用Viterbi思想。选择其中联合概率最大的一条
                    if markov.mid_is_exists(mid):
                        # 计算联合概率。上一个节点到当前节点的联合概率
                        if is_end_word:
                            prob_product = markov.get(last_state_mid).prob_product * prob
                        else:
                            launch_prob = self.hmm_dict.get_launch_prob(word, state)
                            prob_product = markov.get(last_state_mid).prob_product * prob * launch_prob
                        # 对比联合概率大小。如果小于该概率，则更新路径
                        if markov.get(mid).prob_product < prob_product:
                            # 更新路径。保存概率最大的路径
                            markov.get(mid).transition_prob = {last_state_mid: prob}
                            # 更新到该节点的联合概率
                            markov.get(mid).prob_product = prob_product
                    else:
                        # 保存上一个节点到目前节点的转移概率。
                        transition_prob = {last_state_mid: prob}
                        # 如果上一个节点是 (START) , 其联合概率只有当前节点的转移概率和发射概率
                        if markov.get(last_state_mid).state == self.hmm_dict.START_TAG:
                            launch_prob = self.hmm_dict.get_launch_prob(word, state)
                            prob_product = prob * launch_prob
                        else:
                            if is_end_word:
                                # 如果是结束(End)字符。则没有发射概率
                                prob_product = markov.get(last_state_mid).prob_product * prob
                            else:
                                # 联合概率，上一个节点的概率积 * 当前发射概率 * 当前转移概率。
                                launch_prob = self.hmm_dict.get_launch_prob(word, state)
                                prob_product = markov.get(last_state_mid).prob_product * prob * launch_prob
                        # 保存当前节点
                        markov.append(mid=mid, state=state, word=word, launch_prob=launch_prob,
                                      transition_prob=transition_prob, prob_product=prob_product)
                        # 保存当前新节点的 mid
                        save_state_mid.append(mid)

            # 将上一轮次保存的最后状态值赋值给 save_last_state_mid, 重新进入新的轮次
            save_last_state_mid = list(set(save_state_mid))

        # 如果最后到 (End) 节点的 mid 有多个，则报错
        if len(save_last_state_mid) != 1:
            raise UnknownError("Please check code ")

        last_state_mid = save_last_state_mid[0]

        # 回溯。根据最后保存的状态mid。依次反向找到最佳路径
        back_state = list()
        for i in range(len(text) + 1):
            state_node = markov.get(last_state_mid)

            # 保存当前状态
            back_state.append(state_node.state)

            # 如果到 (Start) 则结束
            if state_node.state == self.hmm_dict.START_TAG:
                break

            # 如果某个节点有多个转移概率，则报错
            if len(state_node.transition_prob) != 1:
                raise UnknownError("Please check code ")

            # 更新 mid ，保存上一个 mid 值
            last_state_mid = tuple(state_node.transition_prob.keys())[0]

        # 去掉 (Start) 和 (End) 结束符
        back_state = back_state[1:-1]

        # 反转编码。获得标签
        label = [back_state[-i] for i in range(1, len(back_state) + 1)]

        return label

    def eval(self, dataset, seg_lab="  ", w_file="test.txt", encoding="utf-8", threads=3):
        """
        评估。采用线程并非机制，读取测试数据集，并进行分词
        :param dataset: (str, mandatory) 测试数据集
        :param seg_lab: (str, optional, default="  ") 分词标记。在分词完成之后，需要根据某个字符标记词汇，默认为 " "
        :param w_file: (str, optional, default="test.txt") 分词结果写入的文件名
        :param encoding: (str, optional, default="utf-8") 分词结果写入文件的编码格式
        :param threads: (int, optional, default=3) 启动线程数量，默认为 3
        :return:
        """

        def handle(data, q):
            result = list()
            for d in data:
                # text_list = list(d)
                # label = self.__viterbi(text_list)
                # words = self.hmm_dict.decoding(text_list, label)
                # result.append(seg_lab.join(words))
                result.append(self.cut(d, seg_lab))
                q.put(1)

            return result

        multi_threads = MultiThreading(threads=threads)
        results = multi_threads.process(dataset, handle)

        Writer.writer_file(w_file, results, encoding=encoding)

    def cut(self, data, seg_lab=' '):
        """
        分词。
        :param data: (str, mandatory) 文本。例如"小明是中国人"
        :param seg_lab: (str, optional, default=" ") 分词标记，分词完成之后使用该标记，进行区分。默认问 ' '
        :return: (str) 分词文本
        """
        data_list = list()
        for word in list(data.strip()):
            if word == '' or word == ' ' or len(word) == 0:
                continue
            data_list.append(word)
        label = self.__viterbi(data_list)
        words = self.hmm_dict.decoding(data_list, label)

        return seg_lab.join(words)

    @staticmethod
    def load(model_file):
        """
        加载模型
        :param model_file: (str, mandatory) 模型文件
        :return: (HmmSegmentation) 分词模型
        """
        hmm_seg = HmmSegmentation()
        hmm_seg.hmm_dict = hmm_seg.hmm_dict.load(model_file)

        return hmm_seg


def test_module_func():
    train_dataset = [['这', '首先', '是', '个', '民族', '问题', '，', '的', '感情', '。'],
                     ['我', '扔', '了', '两颗', '手榴弹', '，', '他', '一下子', '出', '溜', '下去', '。'],
                     ['他', '要', '与', '中国人', '合作', '。']
                     ]
    test_dataset = ['他要与中国人合作。']

    model_file = "./hmm_segmentation.pickle"

    hmm_seg = HmmSegmentation()
    hmm_seg.fit(train_dataset, save_model=model_file)
    hmm_seg.eval(test_dataset, seg_lab="  ", w_file="./test.utf8", threads=1)

    sent = "他要与中国人合作。"
    cut_sent = hmm_seg.cut(sent)
    print(hmm_seg.cut(sent))

    load_hmm_seg = HmmSegmentation.load(model_file)
    load_cut_sent = load_hmm_seg.cut(sent)
    assert cut_sent == load_cut_sent, "{} Error: load model segmentation error ".format(load_hmm_seg.cut.__name__)


if __name__ == "__main__":
    test_module_func()
