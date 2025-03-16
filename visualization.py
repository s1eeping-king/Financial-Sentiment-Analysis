import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_history(train_losses, val_accuracies):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/training_metrics/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/model_analysis/confusion_matrix.png')
    plt.close()

def plot_metrics_report(y_true, y_pred, classes):
    """绘制分类报告"""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:-3, :].astype(float), annot=True, cmap='Blues')
    plt.title('Classification Metrics')
    plt.savefig('outputs/model_analysis/metrics_report.png')
    plt.close()

def plot_learning_rate(learning_rates):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates)
    plt.title('Learning Rate over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.savefig('outputs/training_metrics/learning_rate.png')
    plt.close()

def plot_sentiment_distribution(predictions):
    """绘制情感分布"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=predictions)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig('outputs/prediction_results/sentiment_distribution.png')
    plt.close()

def plot_confidence_histogram(confidences, predictions):
    """绘制预测置信度直方图"""
    plt.figure(figsize=(10, 6))
    for sentiment in np.unique(predictions):
        conf = [c for c, p in zip(confidences, predictions) if p == sentiment]
        plt.hist(conf, bins=20, alpha=0.5, label=f'Sentiment {sentiment}')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('outputs/prediction_results/confidence_distribution.png')
    plt.close() 