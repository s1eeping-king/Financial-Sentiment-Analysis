# 金融情绪分析项目

本项目旨在进行金融情绪分析，基于 **DistilBERT** 模型。我们使用金融句子数据集对模型进行训练，并进行情绪预测。

## 项目概述

该项目实现了一个金融文本情感分析系统，可以将金融相关的文本分类为积极(positive)、消极(negative)或中性(neutral)。项目使用了轻量级的 DistilBERT 模型作为基础，并进行了针对金融领域的优化。

## 主要特性

- 基于 DistilBERT 的情感分析模型
- 支持多种损失函数（交叉熵、Focal Loss、标签平滑等）
- 实时训练可视化
- 早停机制防止过拟合
- 详细的预测结果分析和可视化
- 模型性能评估和基准测试

## 获取项目

克隆此项目：

```bash
git clone https://github.com/s1eeping-king/Financial-Sentiment-Analysis.git
```

## 配置说明

在 `train.py` 中的 CONFIG 字典可以配置以下参数：

```python
CONFIG = {
    'loss_function': 'cross_entropy',  # 损失函数类型
    'learning_rate': 2e-5,            # 学习率
    'batch_size': 128,                # 批次大小
    'epochs': 10,                     # 训练轮数
    'use_early_stopping': False,      # 是否启用早停
    # ... 其他配置项
}
```

## 使用方法

1. 准备数据：
   - 将数据集放在 `FinancialPhraseBank` 目录下
   - 支持 CSV 格式，需包含文本和标签列

2. 训练模型：
```bash
python train.py
```

3. 查看结果：
   - 训练过程可视化及不同超参数组合的性能比较：`outputs/visualization_results/`
   - 前100预测结果分析：`outputs/visualization_results/test_predictions.png`
   - 性能评估报告：`outputs/benchmark_results/`
   - 额外可视化分析：
       - `outputs/model_analysis/`
       - `outputs/prediction_results/`
       - `outputs/training_metrics/`

## 可视化功能

项目提供了丰富的可视化功能：

1. 训练过程可视化：
   - 训练损失曲线
   - 训练准确率曲线
   - 验证准确率曲线

2. 预测结果分析：
   - 前100个测试样本的详细预测结果
   - 包含输入文本、真实标签、预测标签和置信度
   - 错误预测标注

## 性能优化

1. 早停机制：
   - 可配置监控指标（验证准确率或训练损失）
   - 可设置等待轮数和最小改善量
   - 自动保存最佳模型

2. 损失函数选择：
   - 交叉熵损失
   - Focal Loss
   - 标签平滑
   - 加权交叉熵
   - Dice Loss
