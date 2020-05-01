## 简述

中文分词模型项目。使用目前比较常见的分词方法，即词典分词（也叫机械分词）、ngram 分词、隐马尔可夫分词、CRF 分词方法。

### 数据集

使用 [icwb-data2](http://sighan.cs.uchicago.edu/bakeoff2005/) 数据集。来自SIGHAN，SIGHAN 是国际计算语言协会ACL中文处理小组的简称。目前SIGHAN bakeoff 已经举办了 6 届，其中语料资源免费。选用 icwb-data2 数据作为数据集。

● icwb-data2 中包含train、test、scripts、gold、doc 目录

```
  ○ doc：数据集的一些使用指南
  ○ training： 包含已经分词的训练数据集目录。这里选择 msr_training.utf8 作为训练集。其他信息可见doc目录下的说明
      ■ 文件后缀名为 utf8 的表示编码格式为 UTF-8。
      ■ 文件前缀 msr_ ，代表是微软亚洲研究院提供。
  ○ testing：未切分的测试数据集
  ○ scripts：评分脚本和简单的分词器
  ○ gold：测试数据集的标准分词和训练集中抽取的词表
```

### 分词评估

| 模型名称   | RECALL | PRECISION | OOV Recall Rate | IV Recall Rate |
| ---------- | ------ | --------- | --------------- | -------------- |
| 词典分词   | 0.957  | 0.917     | 0.025           | 0.983          |
| ngram 分词 | 0.963  | 0.924     | 0.025           | 0.988          |
| HMM 分词   | 0.784  | 0.772     | 0.363           | 0.796          |
| CRF 分词   | 0.850  | 0.852     | 0.587           | 0.857          |

### 使用

模型的使用方法参考：notebook/chinese word segmentation.ipynb 

个人云笔记：https://note.youdao.com/ynoteshare1/index.html?id=81e18f06e7c39d59da74323cc5aff346&type=note