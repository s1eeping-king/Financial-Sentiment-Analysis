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
    if epoch != 'final_test' and epoch != 'final_val':
        return accuracy, (macro_p, macro_r, macro_f1)
    
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算详细指标
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # 获取每个类别的指标
    class_names = ['positive', 'negative', 'neutral']
    
    # 创建表格内容
    lines = []
    lines.append(f"Classification Metrics - {'Test Set' if epoch == 'final_test' else 'Final Validation'}")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # 保存到文件
    filename = os.path.join(save_dir, 'test_metrics.txt' if epoch == 'final_test' else 'final_validation_metrics.txt')
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    
    if epoch == 'final_test':
        print(f"测试集评估完成，结果已保存到: {filename}")
    
    return accuracy, (macro_p, macro_r, macro_f1)
