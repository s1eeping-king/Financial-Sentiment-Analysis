import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class TrainingVisualizer:
    def __init__(self, output_dir, param_name=None, param_value=None):
        """
        初始化训练可视化器
        
        Args:
            output_dir: 输出目录的路径
            param_name: 参数名称（如'learning_rate'）
            param_value: 参数值
        """
        self.output_dir = os.path.join(output_dir, 'visualization_results')
        self.metrics_dir = os.path.join(self.output_dir)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 存储训练指标
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.current_epoch = 0
        
        # 存储参数信息
        self.param_name = param_name
        self.param_value = param_value
        
        # 用于存储多个训练结果的比较
        self.all_results = {}
        
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
        
        # 根据参数信息生成文件名
        if self.param_name and self.param_value is not None:
            filename = f'training_metrics_{self.param_name}_{self.param_value}.png'
        else:
            filename = 'training_metrics.png'
        
        # 保存图像
        save_path = os.path.join(self.metrics_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练指标图表已保存到: {save_path}")
        
    def add_training_result(self, param_name, param_value, train_losses, train_accuracies, test_accuracies):
        """
        添加一组训练结果用于比较
        
        Args:
            param_name: 参数名称（如'learning_rate'）
            param_value: 参数值
            train_losses: 训练损失列表
            train_accuracies: 训练精确度列表
            test_accuracies: 测试精确度列表
        """
        key = f"{param_name}_{param_value}"
        self.all_results[key] = {
            'param_name': param_name,
            'param_value': param_value,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    def plot_comparison_metrics(self):
        """绘制不同参数设置下的训练指标比较图"""
        if not self.all_results:
            print("没有可比较的训练结果")
            return
        
        # 获取所有结果的最大epoch数
        max_epochs = max([len(result['train_losses']) for result in self.all_results.values()])
        epochs = range(1, max_epochs + 1)
        
        # 创建三个图表，分别用于损失、训练精确度和验证精确度
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # 颜色列表，用于区分不同参数
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        # 绘制每个参数的结果
        for i, (key, result) in enumerate(self.all_results.items()):
            color = colors[i % len(colors)]
            param_name = result['param_name']
            param_value = result['param_value']
            label = f"{param_name}={param_value}"
            
            # 确保数据长度一致
            train_losses = result['train_losses'] + [None] * (max_epochs - len(result['train_losses']))
            train_accuracies = result['train_accuracies'] + [None] * (max_epochs - len(result['train_accuracies']))
            test_accuracies = result['test_accuracies'] + [None] * (max_epochs - len(result['test_accuracies']))
            
            # 绘制训练损失
            valid_epochs = [e for e, loss in enumerate(train_losses, 1) if loss is not None]
            valid_losses = [loss for loss in train_losses if loss is not None]
            ax1.plot(valid_epochs, valid_losses, f'{color}-', label=label)
            
            # 绘制训练精确度
            valid_epochs = [e for e, acc in enumerate(train_accuracies, 1) if acc is not None]
            valid_accuracies = [acc for acc in train_accuracies if acc is not None]
            ax2.plot(valid_epochs, valid_accuracies, f'{color}-', label=label)
            
            # 绘制验证精确度
            valid_epochs = [e for e, acc in enumerate(test_accuracies, 1) if acc is not None]
            valid_accuracies = [acc for acc in test_accuracies if acc is not None]
            ax3.plot(valid_epochs, valid_accuracies, f'{color}-', label=label)
        
        # 设置图表1（训练损失）
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss Comparison for Different {param_name} Values')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 设置图表2（训练精确度）
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Training Accuracy Comparison for Different {param_name} Values')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 设置图表3（验证精确度）
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'Validation Accuracy Comparison for Different {param_name} Values')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 保存图表
        plt.figure(fig1.number)
        save_path1 = os.path.join(self.metrics_dir, f'comparison_train_loss_{param_name}.png')
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        
        plt.figure(fig2.number)
        save_path2 = os.path.join(self.metrics_dir, f'comparison_train_accuracy_{param_name}.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        
        plt.figure(fig3.number)
        save_path3 = os.path.join(self.metrics_dir, f'comparison_validation_accuracy_{param_name}.png')
        plt.savefig(save_path3, dpi=300, bbox_inches='tight')
        
        plt.close('all')
        
        print(f"比较图表已保存到:")
        print(f"├── {save_path1}")
        print(f"├── {save_path2}")
        print(f"└── {save_path3}")
        
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

    def visualize_test_predictions(self, texts, true_labels, predicted_labels, confidences, label_names=['positive', 'negative', 'neutral'], n_samples=100, suffix=''):
        """
        可视化测试集的预测结果
        
        Args:
            texts: 输入文本列表
            true_labels: 真实标签列表
            predicted_labels: 预测标签列表
            confidences: 预测置信度列表
            label_names: 标签名称列表
            n_samples: 要显示的样本数量
            suffix: 文件名后缀，用于区分不同参数的结果
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
        
        # 生成文件名
        if self.param_name and self.param_value is not None:
            filename_base = f'test_predictions_{self.param_name}_{self.param_value}{suffix}'
        else:
            filename_base = f'test_predictions{suffix}'
        
        # 保存图表
        save_path = os.path.join(self.metrics_dir, f'{filename_base}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"测试集预测结果可视化已保存到: {save_path}")
        
        # 同时保存为CSV文件以便更好地查看
        csv_path = os.path.join(self.metrics_dir, f'{filename_base}.csv')
        df.to_csv(csv_path, index=False)
        print(f"测试集预测结果已保存到CSV文件: {csv_path}")
