# 金融情绪分析项目

本项目旨在进行金融情绪分析，基于 **DistilBERT** 模型。我们使用金融句子数据集对模型进行训练，并进行情绪预测。

## 获取项目

克隆此项目：

```bash
git clone https://github.com/s1eeping-king/Financial-Sentiment-Analysis.git
```

## 更新工作v2.1

1.优化了训练时长，时间差不多是之前的一半
2.添加了loss_funtion，在train.py的CONFIG可以进行选择
3.添加了benchmark.py，可以比对最后一次验证和测试的性能
4.添加了早停机制，现在可以更早的结束训练了
5.修改了数据集划分，现在训练集、验证集、测试集的比例是8:1:1
6.修改了原先在验证集上的性能评估，现在是在测试集上进行评估
