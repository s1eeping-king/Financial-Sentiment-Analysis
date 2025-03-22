import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report

class VisualizationManager_extra:
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.subdirs = {
            'training_metrics': os.path.join(output_dir, 'training_metrics'),
            'model_analysis': os.path.join(output_dir, 'model_analysis'),
            'prediction_results': os.path.join(output_dir, 'prediction_results')
        }
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保所有输出目录存在"""
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _save_plot(self, filename, subdir):
        """保存图表到指定子目录"""
        path = os.path.join(self.subdirs[subdir], filename)
        plt.savefig(path)
        plt.close()

    def plot_training_history(self, train_losses, val_accuracies):
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
        self._save_plot('training_history.png', 'training_metrics')

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        self._save_plot('confusion_matrix.png', 'model_analysis')

    def plot_metrics_report(self, y_true, y_pred, classes):
        """绘制分类报告"""
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.iloc[:-3, :].astype(float), annot=True, cmap='Blues')
        plt.title('Classification Metrics')
        self._save_plot('metrics_report.png', 'model_analysis')

    def plot_learning_rate(self, learning_rates):
        """绘制学习率变化曲线"""
        plt.figure(figsize=(10, 4))
        plt.plot(learning_rates)
        plt.title('Learning Rate over Training Steps')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        self._save_plot('learning_rate.png', 'training_metrics')

    def plot_sentiment_distribution(self, predictions):
        """绘制情感分布"""
        plt.figure(figsize=(8, 6))
        sns.countplot(x=predictions)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        self._save_plot('sentiment_distribution.png', 'prediction_results')

    def plot_confidence_histogram(self, confidences, predictions):
        """绘制预测置信度直方图"""
        plt.figure(figsize=(10, 6))
        for sentiment in np.unique(predictions):
            conf = [c for c, p in zip(confidences, predictions) if p == sentiment]
            plt.hist(conf, bins=20, alpha=0.5, label=f'Sentiment {sentiment}')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        self._save_plot('confidence_distribution.png', 'prediction_results')

    def analyze_prediction_results(self, predictions, confidences, true_labels, classes=['positive', 'negative', 'neutral']):
        """
        执行完整的预测结果分析流程，生成所有可视化图表
        
        Args:
            predictions: 模型预测结果列表
            confidences: 预测置信度列表
            true_labels: 真实标签列表
            classes: 类别标签列表，默认为['positive', 'negative', 'neutral']
        """
        try:
            print("开始生成预测结果分析图表...")
            
            print("1. 生成情感分布图...")
            self.plot_sentiment_distribution(predictions)
            
            print("2. 生成置信度直方图...")
            self.plot_confidence_histogram(confidences, predictions)
            
            print("3. 生成混淆矩阵...")
            self.plot_confusion_matrix(true_labels, predictions, classes)
            
            print("4. 生成分类指标报告...")
            self.plot_metrics_report(true_labels, predictions, classes)
            
            print("所有可视化图表生成完成！")
            print(f"图表已保存到以下目录：")
            print(f"├── 情感分布图: {self.subdirs['prediction_results']}")
            print(f"├── 置信度直方图: {self.subdirs['prediction_results']}")
            print(f"├── 混淆矩阵: {self.subdirs['model_analysis']}")
            print(f"└── 分类指标报告: {self.subdirs['model_analysis']}")
            
        except Exception as e:
            print(f"在生成分析图表时出错: {str(e)}")
            import traceback
            traceback.print_exc() 