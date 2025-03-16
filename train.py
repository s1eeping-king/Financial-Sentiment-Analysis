import torch
import os
import time
import numpy as np
from model import SentimentClassifier, prepare_data, predict_sentiment
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import pandas as pd
from transformers import DistilBertTokenizer
import traceback
import warnings
from visualization import (
    plot_training_history, plot_confusion_matrix, plot_metrics_report,
    plot_learning_rate, plot_sentiment_distribution,
    plot_confidence_histogram
)
from loss_function import get_loss_function
from benchmark import save_classification_metrics

# 忽略警告，防止干扰输出
warnings.filterwarnings("ignore")

# Hyperparameters and Configuration
CONFIG = {
    'loss_function': 'focal',

    # 'loss_function': 'cross_entropy',
    # 'loss_function': 'focal',
    # 'loss_function': 'label_smoothing',
    # 'loss_function': 'weighted_cross_entropy',
    # 'loss_function': 'dice',

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
        'training_metrics': 'training_metrics',
        'prediction_results': 'prediction_results',
        'model_analysis': 'model_analysis'
    },
    
    # Debug mode settings
    'debug_mode': False,
    'debug': {
        'epochs': 2,
        'batch_size': 32,
        'train_samples': 100,
        'val_samples': 50,
    }
}

def ensure_output_dirs():
    """确保所有输出目录存在"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for subdir in CONFIG['subdirs'].values():
        path = os.path.join(CONFIG['output_dir'], subdir)
        os.makedirs(path, exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

def get_output_path(category, filename):
    """获取输出文件的完整路径"""
    return os.path.join(CONFIG['output_dir'], CONFIG['subdirs'][category], filename)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model_with_history(train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=2e-5, dropout_rate=0.3):
    """训练模型并记录训练历史"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    model = SentimentClassifier()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = get_loss_function(CONFIG['loss_function'])
    print(f"使用损失函数: {CONFIG['loss_function']}")
    
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
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
        
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        # 如果是最后一轮或即将早停，保存为final_val，否则正常记录
        is_final = early_stopping.counter + 1 >= early_stopping.patience
        epoch_mark = 'final_val' if is_final else epoch + 1
        accuracy, _ = save_classification_metrics(all_labels, all_preds, epoch_mark)
        val_accuracies.append(accuracy)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'平均训练损失: {avg_train_loss:.4f}')
        print(f'验证准确率: {accuracy:.2f}%')
        print(f'本轮训练时间: {epoch_time:.2f}秒\n')
        
        plot_training_history(train_losses, val_accuracies)
        plot_learning_rate(learning_rates)
        
        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            print("触发早停")
            break
    
    return model, train_losses, val_accuracies

def save_model(model, tokenizer, model_dir='saved_model'):
    """保存模型和分词器"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"模型已保存到: {model_path}")
    print(f"分词器已保存到: {tokenizer_path}")

def load_model(model_dir='saved_model'):
    """加载保存的模型和分词器"""
    model = SentimentClassifier()
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer

def predict_with_confidence(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """使用模型进行预测并返回置信度"""
    model.eval()
    encoded = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    return prediction.item(), confidence.item()

def analyze_predictions(model, tokenizer, dataset):
    """分析模型预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    texts = []
    predictions = []
    confidences = []
    true_labels = []
    
    try:
        print("开始收集预测结果...")
        for item in dataset:
            text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
            pred, conf = predict_with_confidence(text, model, tokenizer, device)
            
            texts.append(text)
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(item['labels'].item())
        
        print("生成情感分布图...")
        plot_sentiment_distribution(predictions)
        
        print("生成置信度直方图...")
        plot_confidence_histogram(confidences, predictions)
        
        print("生成混淆矩阵...")
        classes = ['positive', 'negative', 'neutral']
        plot_confusion_matrix(true_labels, predictions, classes)
        
        print("生成分类指标报告...")
        plot_metrics_report(true_labels, predictions, classes)
        
        print("所有可视化图表生成完成")
        
    except Exception as e:
        print(f"在生成分析图表时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return texts, predictions, confidences

def get_attention_weights(text, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """获取注意力权重"""
    model.eval()
    encoded = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        attention = model.bert.transformer.layer[-1].attention.self.attention_weights
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attention_weights = attention[0].mean(dim=0).cpu().numpy()
    
    return attention_weights, tokens

def analyze_feature_importance(model, tokenizer, dataset, n_samples=100):
    """分析特征重要性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    feature_importance = np.zeros(model.bert.config.hidden_size)
    
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
            
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = model.bert.transformer.layer[-1].output
            
        feature_importance += hidden_states.mean(dim=1).abs().cpu().numpy()[0]
    
    feature_importance /= min(n_samples, len(dataset))
    
    return feature_importance

def main():
    """主函数"""
    torch.manual_seed(CONFIG['random_seed'])
    ensure_output_dirs()
    
    data_path = 'FinancialPhraseBank/all-data.csv'
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    print("准备数据...")
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(
        data_path, 
        max_len=CONFIG['max_seq_length']
    )
    
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
    
    print("\n开始训练模型...")
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
    training_time = end_time - start_time
    
    print(f"\n训练完成:")
    print(f"├── 总训练时间: {training_time:.2f}秒")
    print(f"└── 平均每轮时间: {training_time/CONFIG['epochs']:.2f}秒")
    # 在测试集上评估
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    save_classification_metrics(all_labels, all_preds, epoch='final_test', save_dir='outputs/benchmark_results')
    save_model(model, tokenizer)
    
    print("\n开始生成测试集预测结果分析...")
    analyze_predictions(model, tokenizer, test_dataset)  # 改用测试集进行最终分析

if __name__ == "__main__":
    main()