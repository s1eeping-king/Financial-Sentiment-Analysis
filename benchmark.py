from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
import os

def save_classification_metrics(y_true, y_pred, epoch, total_epochs=None, save_dir='outputs/benchmark_results'):
    """保存分类指标到txt文件
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        epoch: 当前epoch或'final_test'或'final_val'
        total_epochs: 总的训练轮数（不再使用）
        save_dir: 保存目录
    """
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # 如果不是最终验证或测试集评估，直接返回指标值
    if not isinstance(epoch, str) or (epoch != 'final_test' and not epoch.startswith('final_test_') and epoch != 'final_val'):
        return accuracy, (macro_p, macro_r, macro_f1)
    
    # 调试信息
    print(f"准备保存分类指标到目录: {save_dir}")
    print(f"当前epoch标识: {epoch}")
    
    # 确保输出目录存在
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"目录创建成功或已存在: {save_dir}")
    except Exception as e:
        print(f"创建目录时出错: {str(e)}")
        # 尝试使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, save_dir)
        print(f"尝试使用绝对路径: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    
    # 计算详细指标
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 获取每个类别的指标
    class_names = ['positive', 'negative', 'neutral']
    
    # 创建表格内容
    lines = []
    lines.append(f"Classification Metrics - {'Test Set' if 'final_test' in epoch else 'Final Validation'}")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Epoch: {epoch}")  # 添加epoch信息
    lines.append("-" * 80)
    
    # 总体准确率
    lines.append(f"Overall Accuracy: {accuracy:.4f}")
    lines.append("-" * 80)
    
    # 表头
    lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    lines.append("-" * 80)
    
    # 每个类别的指标
    for i, class_name in enumerate(class_names):
        lines.append(
            f"{class_name:<15} "
            f"{precision[i]:<12.4f} "
            f"{recall[i]:<12.4f} "
            f"{f1[i]:<12.4f} "
            f"{support[i]:<12d}"
        )
    
    # 添加平均值
    lines.append("-" * 80)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    lines.append(
        f"{'Macro Avg':<15} "
        f"{macro_p:<12.4f} "
        f"{macro_r:<12.4f} "
        f"{macro_f1:<12.4f} "
        f"{sum(support):<12d}"
    )
    lines.append(
        f"{'Weighted Avg':<15} "
        f"{weighted_p:<12.4f} "
        f"{weighted_r:<12.4f} "
        f"{weighted_f1:<12.4f} "
        f"{sum(support):<12d}"
    )
    
    # 生成文件名
    if epoch == 'final_test':
        filename = 'test_metrics.txt'
    elif epoch.startswith('final_test_'):
        # 从epoch中提取参数信息
        param_info = epoch.replace('final_test_', '')
        filename = f'test_metrics_{param_info}.txt'
    else:
        filename = 'final_validation_metrics.txt'
    
    full_path = os.path.join(save_dir, filename)
    print(f"将保存指标到文件: {full_path}")
    
    # 保存到文件
    try:
        with open(full_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"成功保存指标到文件: {full_path}")
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
    
    return accuracy, (macro_p, macro_r, macro_f1)
