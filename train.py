import torch
import os
import time
import numpy as np
from model import SentimentClassifier, prepare_data, predict_sentiment
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime
import pandas as pd
from transformers import DistilBertTokenizer
import traceback
import warnings
from visualization import TrainingVisualizer
from visualization_extra import VisualizationManager_extra
from loss_function import get_loss_function
from benchmark import save_classification_metrics
from early_stopping import EarlyStopping

# 忽略警告，防止干扰输出
warnings.filterwarnings("ignore")

# Hyperparameters and Configuration
CONFIG = {
    'loss_function': 'cross_entropy',

    # 'loss_function': 'cross_entropy',
    # 'loss_function': 'focal',
    # 'loss_function': 'label_smoothing',
    # 'loss_function': 'weighted_cross_entropy',
    # 'loss_function': 'dice',

    'data_path': 'FinancialPhraseBank/Sentences_AllAgree.csv',
    # 'data_path': 'FinancialPhraseBank/all-data.csv',
    # 'data_path': 'FinancialPhraseBank/forTest.csv',

    # Model hyperparameters
    'learning_rate': 2e-5,
    'dropout_rate': 0.3,
    'max_seq_length': 315,
    
    # Training hyperparameters
    'batch_size': 64,
    'epochs': 10,
    'train_split_ratio': 0.8,
    
    # Early stopping settings
    'use_early_stopping': False,  # 是否启用早停
    'early_stopping': {
        'patience': 3,           # 等待改善的轮数
        'mode': 'max',          # 监控指标的模式 ('min' 或 'max')
        'min_delta': 0.001,     # 最小改善量
        'monitor': 'val_acc'    # 监控的指标 ('val_acc' 或 'train_loss')
    },
    
    # Other settings
    'random_seed': 42,
    'model_dir': 'saved_model',
    
    # Output directories
    'output_dir': 'outputs',
    'subdirs': {
        'training_metrics': 'training_metrics',
        'prediction_results': 'prediction_results',
        'model_analysis': 'model_analysis'
    }
}

def ensure_output_dirs():
    """确保所有输出目录存在"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for subdir in CONFIG['subdirs'].values():
        path = os.path.join(CONFIG['output_dir'], subdir)
        os.makedirs(path, exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

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
    
    viz = TrainingVisualizer(CONFIG['output_dir'])
    
    # 初始化早停机制
    early_stopper = None
    if CONFIG['use_early_stopping']:
        es_config = CONFIG['early_stopping']
        early_stopper = EarlyStopping(
            patience=es_config['patience'],
            mode=es_config['mode'],
            min_delta=es_config['min_delta'],
            verbose=True
        )
        print(f"启用早停机制: patience={es_config['patience']}, mode={es_config['mode']}, "
              f"min_delta={es_config['min_delta']}, monitor={es_config['monitor']}")
    
    best_model_state = None
    best_epoch = -1
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        train_predictions = []
        train_true_labels = []
        
        # 训练阶段
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
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        # 验证阶段
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        # 更新可视化数据
        viz.update_metrics(avg_train_loss, train_accuracy, val_accuracy)
        viz.plot_training_metrics()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'平均训练损失: {avg_train_loss:.4f}')
        print(f'训练准确率: {train_accuracy*100:.2f}%')
        print(f'验证准确率: {val_accuracy*100:.2f}%')
        print(f'本轮训练时间: {epoch_time:.2f}秒\n')
        
        # 早停检查
        if CONFIG['use_early_stopping']:
            monitor_value = val_accuracy if CONFIG['early_stopping']['monitor'] == 'val_acc' else -avg_train_loss
            if early_stopper(epoch, monitor_value):
                print(f"触发早停! 最佳性能出现在epoch {early_stopper.get_best_epoch() + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            elif early_stopper.get_best_epoch() == epoch:
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
    
    if best_epoch >= 0:
        print(f"使用最佳模型（来自epoch {best_epoch + 1}）")
    
    return model

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
    viz_manager = VisualizationManager_extra(CONFIG['output_dir'])
    
    try:
        print("开始收集预测结果...")
        for item in dataset:
            text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
            pred, conf = predict_with_confidence(text, model, tokenizer, device)
            
            texts.append(text)
            predictions.append(pred)
            confidences.append(conf)
            true_labels.append(item['labels'].item())
        
        # 使用新的分析方法
        classes = ['positive', 'negative', 'neutral']
        viz_manager.analyze_prediction_results(predictions, confidences, true_labels, classes)
        
    except Exception as e:
        print(f"在分析预测结果时出错: {str(e)}")
        traceback.print_exc()
    
    return texts, predictions, confidences

def main():
    """主函数"""
    torch.manual_seed(CONFIG['random_seed'])
    ensure_output_dirs()
    
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    print("准备数据...")
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(
        data_path, 
        max_len=CONFIG['max_seq_length']
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    print("\n开始训练模型...")
    start_time = time.time()
    
    model = train_model_with_history(
        train_dataset, 
        val_dataset, 
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n训练完成:")
    print(f"├── 总训练时间: {training_time:.2f}秒")
    print(f"└── 平均每轮时间: {training_time/CONFIG['epochs']:.2f}秒")
    
    print("\n开始在测试集上评估...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    all_texts = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取原始文本
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            all_texts.extend(texts)
            
            # 获取预测结果和置信度
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    # 保存分类指标
    save_classification_metrics(all_labels, all_preds, epoch='final_test', save_dir='outputs/benchmark_results')
    
    # 可视化测试集预测结果
    viz = TrainingVisualizer(CONFIG['output_dir'])
    viz.visualize_test_predictions(
        texts=all_texts,
        true_labels=all_labels,
        predicted_labels=all_preds,
        confidences=all_confidences
    )
    
    save_model(model, tokenizer)
    
    print("\n开始生成测试集预测结果分析...")
    analyze_predictions(model, tokenizer, test_dataset)  # 改用测试集进行最终分析

if __name__ == "__main__":
    main()