import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class TrainingVisualizer:
    def __init__(self, output_dir):
        """
        初始化训练可视化器
        
        Args:
            output_dir: 输出目录的路径
        """
        self.output_dir = os.path.join(output_dir, 'visualization_results')
        self.metrics_dir = os.path.join(self.output_dir)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 存储训练指标
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.current_epoch = 0
        
    def update_metrics(self, train_loss, train_accuracy, test_accuracy):
        """
        更新训练指标
        
        Args:
            train_loss: 当前epoch的平均训练损失
            train_accuracy: 当前epoch的训练精确度
            test_accuracy: 当前epoch的测试精确度
        """
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        self.current_epoch += 1
        
    def plot_training_metrics(self):
        """绘制训练指标随epoch变化的曲线"""
        epochs = range(1, self.current_epoch + 1)
        
        # 创建图表和子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左侧子图：训练损失
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss vs Epochs')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右侧子图：训练和验证精确度
        ax2.plot(epochs, self.train_accuracies, 'g-', label='Training Accuracy')
        ax2.plot(epochs, self.test_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy vs Epochs')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.metrics_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练指标图表已保存到: {save_path}")
        
    def get_current_metrics(self):
        """获取最新的训练指标"""
        if self.current_epoch > 0:
            return {
                'epoch': self.current_epoch,
                'train_loss': self.train_losses[-1],
                'train_accuracy': self.train_accuracies[-1],
                'test_accuracy': self.test_accuracies[-1]
            }
        return None

    def visualize_test_predictions(self, texts, true_labels, predicted_labels, confidences, label_names=['positive', 'negative', 'neutral'], n_samples=100):
        """
        可视化测试集的预测结果
        
        Args:
            texts: 输入文本列表
            true_labels: 真实标签列表
            predicted_labels: 预测标签列表
            confidences: 预测置信度列表
            label_names: 标签名称列表
            n_samples: 要显示的样本数量
        """
        # 确保只取前n_samples个样本
        n_samples = min(n_samples, len(texts))
        texts = texts[:n_samples]
        true_labels = true_labels[:n_samples]
        predicted_labels = predicted_labels[:n_samples]
        confidences = confidences[:n_samples]
        
        # 创建数据框
        df = pd.DataFrame({
            'Text': texts,
            'True Label': [label_names[label] for label in true_labels],
            'Predicted Label': [label_names[label] for label in predicted_labels],
            'Confidence': confidences
        })
        
        # 添加预测正确/错误的标记
        df['Correct'] = df['True Label'] == df['Predicted Label']
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 创建表格
        table = plt.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='left',
            loc='center',
            cellColours=[['white' if correct else '#ffcccc' for _ in range(len(df.columns))] for correct in df['Correct']],
            colColours=['#e6e6e6'] * len(df.columns)
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 隐藏坐标轴
        plt.axis('off')
        
        # 添加标题
        plt.title('')
        
        # 保存图表
        save_path = os.path.join(self.metrics_dir, 'test_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"测试集预测结果可视化已保存到: {save_path}")
        
        # 同时保存为CSV文件以便更好地查看
        csv_path = os.path.join(self.metrics_dir, 'test_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"测试集预测结果已保存到CSV文件: {csv_path}")
