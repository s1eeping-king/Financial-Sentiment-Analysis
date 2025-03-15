import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from model import SentimentClassifier, prepare_data, predict_sentiment
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from transformers import DistilBertTokenizer
import matplotlib.patches as patches
import traceback
import warnings

# 忽略警告，防止干扰输出
warnings.filterwarnings("ignore")

# 下载必要的NLTK数据
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"注意: NLTK数据下载失败，但将继续执行: {str(e)}")

# Hyperparameters and Configuration
CONFIG = {
    # Model hyperparameters
    'learning_rate': 2e-5,
    'dropout_rate': 0.3,
    'max_seq_length': 315,
    
    # Training hyperparameters
    'batch_size': 64,
    'epochs': 10,
    'train_split_ratio': 0.8,
    
    # Other settings
    'random_seed': 42,
    'model_dir': 'saved_model',
    
    # Output directories
    'output_dir': 'outputs',
    'subdirs': {
        'training_metrics': 'training_metrics',  # 训练过程的指标
        'prediction_results': 'prediction_results',  # 预测结果的可视化
        'model_analysis': 'model_analysis'  # 模型分析的可视化
    },
    
    # Debug mode settings
    'debug_mode': False,  # 设置为True启用调试模式
    'debug': {
        'epochs': 2,  # 调试模式下的训练轮次
        'batch_size': 32,  # 调试模式下的批次大小
        'train_samples': 100,  # 调试模式下使用的训练样本数
        'val_samples': 50,  # 调试模式下使用的验证样本数
    }
}

def ensure_output_dirs():
    """确保所有输出目录存在"""
    # 创建主输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 创建子目录
    for subdir in CONFIG['subdirs'].values():
        path = os.path.join(CONFIG['output_dir'], subdir)
        os.makedirs(path, exist_ok=True)
    
    # 创建模型保存目录
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

def get_output_path(category, filename):
    """获取输出文件的完整路径"""
    return os.path.join(CONFIG['output_dir'], CONFIG['subdirs'][category], filename)

def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(CONFIG['random_seed'])
    
    # 数据文件路径
    data_path = 'FinancialPhraseBank/all-data.csv'
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    # 准备数据
    print("正在准备数据...")
    train_dataset, val_dataset, tokenizer = prepare_data(
        data_path, 
        max_len=CONFIG['max_seq_length'],
        train_split_ratio=CONFIG['train_split_ratio']
    )
    
    # 如果是调试模式，只使用部分数据
    if CONFIG['debug_mode']:
        print("\n调试模式已启用:")
        print(f"├── 使用 {CONFIG['debug']['train_samples']} 个训练样本")
        print(f"├── 使用 {CONFIG['debug']['val_samples']} 个验证样本")
        print(f"├── 训练轮次: {CONFIG['debug']['epochs']}")
        print(f"└── 批次大小: {CONFIG['debug']['batch_size']}\n")
        
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(min(len(train_dataset), CONFIG['debug']['train_samples']))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, 
            range(min(len(val_dataset), CONFIG['debug']['val_samples']))
        )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 训练模型
    print("\n开始训练模型...")
    
    # 开始基准测试
    start_time = time.time()
    model, train_losses, val_accuracies = train_model_with_history(
        train_dataset, 
        val_dataset, 
        epochs=CONFIG['debug']['epochs'] if CONFIG['debug_mode'] else CONFIG['epochs'],
        batch_size=CONFIG['debug']['batch_size'] if CONFIG['debug_mode'] else CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        dropout_rate=CONFIG['dropout_rate']
    )
    end_time = time.time()
    
    # 输出训练时间
    training_time = end_time - start_time
    print(f"\n训练完成:")
    print(f"├── 总训练时间: {training_time:.2f}秒")
    print(f"├── 平均每轮时间: {training_time/CONFIG['epochs']:.2f}秒")
    
    # 保存模型
    save_model(model, tokenizer)
    
    # 可视化训练过程
    plot_training_history(train_losses, val_accuracies)
    
    # 生成预测结果可视化
    print("\n开始生成预测结果和模型分析可视化...")
    texts, predictions, confidences = analyze_predictions(model, tokenizer, val_dataset)
    
    # 执行模型分析
    errors, feature_importance = perform_model_analysis(model, tokenizer, val_dataset)
    
    # 输出一些示例预测结果
    print("\n示例预测结果:")
    for i in range(min(5, len(texts))):
        print(f"├── 文本: {texts[i][:100]}...")
        print(f"│   预测: {predictions[i]} (置信度: {confidences[i]:.2f})")


def train_model_with_history(train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=2e-5, dropout_rate=0.3):
    """扩展训练函数，记录训练历史"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    model = SentimentClassifier()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 用于记录训练历史
    train_losses = []
    val_accuracies = []
    epoch_times = []
    learning_rates = []
    all_predictions = []
    all_labels = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            learning_rates.append(optimizer.param_groups[0]['lr'])
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        epoch_predictions = []
        epoch_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 收集预测结果和真实标签
                epoch_predictions.extend(predicted.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
        
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        # 记录每个epoch的预测结果
        all_predictions.extend(epoch_predictions)
        all_labels.extend(epoch_labels)
        
        # 记录训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'平均训练损失: {avg_train_loss:.4f}')
        print(f'验证准确率: {val_accuracy:.2f}%')
        print(f'本轮训练时间: {epoch_time:.2f}秒\n')
    
    # 绘制各种可视化图表
    classes = ['positive', 'negative', 'neutral']
    plot_confusion_matrix(all_labels, all_predictions, classes)
    plot_metrics_report(all_labels, all_predictions, classes)
    plot_learning_rate(learning_rates)
    plot_training_time(epoch_times)
    
    # 保存训练指标到CSV文件
    metrics_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'val_accuracy': val_accuracies,
        'training_time': epoch_times
    })
    metrics_df.to_csv(get_output_path('training_metrics', 'training_metrics.csv'), index=False)
    
    return model, train_losses, val_accuracies

def save_model(model, tokenizer, model_dir='saved_model'):
    """保存模型和分词器"""
    # 创建保存目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'sentiment_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 保存分词器
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    print(f"分词器已保存到 {tokenizer_path}")

def load_model(model_dir='saved_model'):
    """加载保存的模型和分词器"""
    from transformers import DistilBertTokenizer
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 找不到模型目录 {model_dir}")
        return None, None
    
    # 加载分词器
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    if os.path.exists(tokenizer_path):
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"警告: 找不到保存的分词器，使用预训练分词器")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 加载模型
    model_path = os.path.join(model_dir, 'sentiment_model.pth')
    if os.path.exists(model_path):
        model = SentimentClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"模型已从 {model_path} 加载")
        return model, tokenizer
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return None, None

def plot_training_history(train_losses, val_accuracies):
    """可视化训练历史"""
    epochs = range(1, len(train_losses) + 1)
    
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制训练损失
    ax1.plot(epochs, train_losses, 'b-', marker='o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 绘制验证准确率
    ax2.plot(epochs, val_accuracies, 'r-', marker='o')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'training_history.png'))
    print("训练历史图表已保存为 'training_history.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'confusion_matrix.png'))
    plt.close()

def plot_metrics_report(y_true, y_pred, classes):
    """绘制分类指标报告"""
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']].iloc[:-3],
                annot=True, cmap='RdYlGn')
    plt.title('Classification Metrics')
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'classification_metrics.png'))
    plt.close()

def plot_learning_rate(learning_rates):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, marker='o')
    plt.title('Learning Rate over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'learning_rate.png'))
    plt.close()

def plot_training_time(times):
    """绘制训练时间统计"""
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(times)), times)
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'training_time.png'))
    plt.close()

def predict_with_confidence(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """预测情感并返回置信度"""
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    return sentiment_map[predicted.item()], confidence.item()

def plot_sentiment_distribution(predictions):
    """绘制情感分布饼图"""
    sentiment_counts = Counter(predictions)
    plt.figure(figsize=(10, 8))
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    plt.savefig(get_output_path('prediction_results', 'sentiment_distribution.png'))
    plt.close()

def plot_confidence_histogram(confidences, predictions):
    """绘制置信度分布直方图"""
    plt.figure(figsize=(12, 6))
    for sentiment in ['positive', 'negative', 'neutral']:
        conf = [c for c, p in zip(confidences, predictions) if p == sentiment]
        plt.hist(conf, bins=20, alpha=0.5, label=sentiment)
    plt.title('Confidence Distribution by Sentiment')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(get_output_path('prediction_results', 'confidence_distribution.png'))
    plt.close()

def create_wordcloud_by_sentiment(texts, predictions, sentiment_filter):
    """为特定情感创建词云"""
    # 过滤出指定情感的文本
    filtered_texts = [text for text, pred in zip(texts, predictions) if pred == sentiment_filter]
    if not filtered_texts:
        return
    
    # 文本预处理
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    
    # 合并所有文本并分词
    words = []
    for text in filtered_texts:
        tokens = tokenizer.tokenize(text.lower())
        words.extend([w for w in tokens if w not in stop_words and len(w) > 2])
    
    # 创建词云
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment_filter.capitalize()} Sentiment WordCloud')
    plt.savefig(get_output_path('prediction_results', f'wordcloud_{sentiment_filter}.png'))
    plt.close()

def extract_and_plot_keywords(texts, predictions):
    """提取和可视化关键词"""
    sentiment_words = {
        'positive': Counter(),
        'negative': Counter(),
        'neutral': Counter()
    }
    
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    
    # 按情感分类统计词频
    for text, sentiment in zip(texts, predictions):
        words = tokenizer.tokenize(text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        sentiment_words[sentiment].update(words)
    
    # 绘制每种情感的top关键词
    plt.figure(figsize=(15, 5))
    for i, (sentiment, counter) in enumerate(sentiment_words.items()):
        top_words = dict(counter.most_common(10))
        
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(top_words)), list(top_words.values()))
        plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
        plt.title(f'Top Keywords - {sentiment.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(get_output_path('prediction_results', 'keywords_distribution.png'))
    plt.close()

def analyze_predictions(model, tokenizer, dataset):
    """分析预测结果并生成可视化"""
    texts = []
    predictions = []
    confidences = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("\n开始生成预测结果可视化...")
    
    # 收集预测结果
    for item in dataset:
        text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        texts.append(text)
        sentiment, confidence = predict_with_confidence(text, model, tokenizer, device)
        predictions.append(sentiment)
        confidences.append(confidence)
    
    # 生成各种可视化
    print("├── 生成情感分布饼图...")
    plot_sentiment_distribution(predictions)
    
    print("├── 生成置信度分布直方图...")
    plot_confidence_histogram(confidences, predictions)
    
    print("├── 生成词云图...")
    for sentiment in ['positive', 'negative', 'neutral']:
        create_wordcloud_by_sentiment(texts, predictions, sentiment)
    
    print("└── 生成关键词分布图...")
    extract_and_plot_keywords(texts, predictions)
    
    return texts, predictions, confidences

def get_attention_weights(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """获取模型的注意力权重"""
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    try:
        with torch.no_grad():
            # 尝试直接访问distilbert层
            outputs = model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取隐藏状态
            hidden_states = outputs[0]  # [batch_size, seq_len, hidden_dim]
            
            # 创建一个简单的自注意力矩阵 (使用隐藏状态的相似度)
            batch_size, seq_len, hidden_dim = hidden_states.size()
            hidden_states_flat = hidden_states.view(batch_size, seq_len, hidden_dim)
            
            # 计算注意力矩阵 (相似度矩阵)
            attention = torch.bmm(hidden_states_flat, hidden_states_flat.transpose(1, 2))
            
            # 应用attention mask
            mask = attention_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
            mask = mask & mask.transpose(1, 2)
            attention = attention.masked_fill_(~mask.bool(), -1e9)
            
            # 使用softmax归一化
            attention = F.softmax(attention, dim=-1)
        
        # 返回第一个样本的注意力矩阵
        return attention[0].cpu().numpy(), encoding
        
    except AttributeError:
        # 如果无法访问model.distilbert，创建一个简化的注意力矩阵
        print("    ├── 注意: 无法访问模型的distilbert层，创建替代注意力矩阵")
        
        with torch.no_grad():
            # 获取序列长度
            seq_len = input_ids.size(1)
            
            # 生成替代的注意力矩阵
            attention = torch.zeros((seq_len, seq_len), device=device)
            
            # 生成一个简单的衰减注意力模式
            for i in range(seq_len):
                attention[i] = torch.exp(-0.1 * torch.abs(torch.arange(seq_len, device=device) - i))
            
            # 应用mask
            mask = attention_mask[0].unsqueeze(0) * attention_mask[0].unsqueeze(1)
            attention = attention.masked_fill(mask == 0, 0)
            
            # 归一化
            row_sums = attention.sum(dim=1, keepdim=True)
            attention = attention / (row_sums + 1e-8)
        
        return attention.cpu().numpy(), encoding

def plot_attention_heatmap(text, model, tokenizer):
    """绘制注意力权重热力图"""
    attention, encoding = get_attention_weights(text, model, tokenizer)
    
    # 获取分词后的文本
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    
    # 过滤掉padding tokens
    attention_mask = encoding['attention_mask'][0].cpu().numpy()
    valid_length = int(attention_mask.sum())
    
    # 只显示非padding的token
    valid_attention = attention[:valid_length, :valid_length]
    valid_tokens = tokens[:valid_length]
    
    # 创建热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(valid_attention, 
                xticklabels=valid_tokens,
                yticklabels=valid_tokens,
                cmap='YlOrRd',
                center=0.5,
                vmin=0.0,
                vmax=1.0)
    plt.title('Token Attention Pattern', fontsize=16)
    plt.xlabel('Target Tokens', fontsize=14)
    plt.ylabel('Source Tokens', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'attention_pattern.png'), dpi=300)
    plt.close()

def analyze_prediction_errors(model, tokenizer, dataset, max_examples=10):
    """分析预测错误的案例"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    errors = []
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    for item in dataset:
        text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        true_label = sentiment_map[item['labels'].item()]
        
        # 获取预测和置信度
        pred_sentiment, confidence = predict_with_confidence(text, model, tokenizer, device)
        
        if pred_sentiment != true_label:
            errors.append({
                'text': text,
                'true_label': true_label,
                'predicted': pred_sentiment,
                'confidence': confidence
            })
            
            if len(errors) >= max_examples:
                break
    
    # 创建错误分析报告
    report = pd.DataFrame(errors)
    report.to_csv(get_output_path('model_analysis', 'prediction_errors.csv'), index=False)
    
    # 可视化错误案例的置信度分布
    plt.figure(figsize=(10, 6))
    for sentiment in ['positive', 'negative', 'neutral']:
        conf = [e['confidence'] for e in errors if e['true_label'] == sentiment]
        if conf:
            plt.hist(conf, bins=10, alpha=0.5, label=f'True {sentiment}')
    
    plt.title('Confidence Distribution of Misclassified Examples')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(get_output_path('model_analysis', 'error_confidence_dist.png'))
    plt.close()
    
    return errors

def analyze_feature_importance(model, tokenizer, dataset, n_samples=100):
    """分析特征重要性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 收集词语重要性得分
    word_importance = Counter()
    processed_samples = 0
    
    for item in dataset:
        if processed_samples >= n_samples:
            break
            
        text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        
        # 使用正则表达式直接分词，避免NLTK的依赖问题
        tokenizer_pattern = RegexpTokenizer(r'\w+')
        words = tokenizer_pattern.tokenize(text)
        
        # 获取原始预测
        original_sentiment, original_conf = predict_with_confidence(text, model, tokenizer, device)
        
        # 对每个词计算重要性
        for i, word in enumerate(words):
            # 创建去掉当前词的文本
            modified_text = ' '.join(words[:i] + words[i+1:])
            modified_sentiment, modified_conf = predict_with_confidence(modified_text, model, tokenizer, device)
            
            # 如果预测发生变化或置信度显著下降，增加该词的重要性分数
            if modified_sentiment != original_sentiment:
                word_importance[word.lower()] += 1
            elif original_conf - modified_conf > 0.1:
                word_importance[word.lower()] += (original_conf - modified_conf)
        
        processed_samples += 1
    
    # 可视化特征重要性
    top_words = dict(word_importance.most_common(20))
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.title('Feature Importance Analysis')
    plt.xlabel('Words')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(get_output_path('model_analysis', 'feature_importance.png'))
    plt.close()
    
    # 保存特征重要性分数
    importance_df = pd.DataFrame(list(word_importance.items()), columns=['word', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(get_output_path('model_analysis', 'feature_importance.csv'), index=False)
    
    return word_importance

def perform_model_analysis(model, tokenizer, dataset):
    """执行完整的模型分析"""
    print("\n开始进行模型分析...")
    errors = None
    feature_importance = None
    
    # 分析注意力模式
    print("├── 生成注意力模式可视化...")
    try:
        example_texts = [
            "This is a very interesting financial product with good potential.",
            "The company's quarterly earnings were disappointing and below expectations.",
            "Market conditions remain stable with no significant changes."
        ]
        for i, text in enumerate(example_texts):
            try:
                plot_attention_heatmap(text, model, tokenizer)
                print(f"    ├── 已生成示例 {i+1} 的注意力热力图")
            except Exception as e:
                print(f"    ├── 生成示例 {i+1} 的注意力热力图时出错: {str(e)}")
    except Exception as e:
        print(f"    └── 注意力可视化过程中出错: {str(e)}")
    
    # 分析预测错误
    print("├── 分析预测错误案例...")
    try:
        errors = analyze_prediction_errors(model, tokenizer, dataset)
        print(f"    └── 已找到 {len(errors)} 个预测错误案例")
    except Exception as e:
        print(f"    └── 分析预测错误时出错: {str(e)}")
    
    # 分析特征重要性
    print("└── 分析特征重要性...")
    try:
        feature_importance = analyze_feature_importance(model, tokenizer, dataset)
        top_features = list(feature_importance.most_common(5))
        print(f"    └── 已分析特征重要性，前5个重要特征: {top_features}")
    except Exception as e:
        print(f"    └── 分析特征重要性时出错: {str(e)}")
    
    return errors, feature_importance

if __name__ == "__main__":
    ensure_output_dirs()
    main()