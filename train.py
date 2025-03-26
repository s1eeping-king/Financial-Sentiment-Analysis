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
    # 'loss_function': ['cross_entropy', 'focal', 'dice', 'label_smoothing', 'weighted_cross_entropy'],
    'loss_function': 'cross_entropy',
    
    # 'data_path': [
    #     'FinancialPhraseBank/Sentences_50Agree.csv',
    #     'FinancialPhraseBank/Sentences_66Agree.csv',
    #     'FinancialPhraseBank/Sentences_75Agree.csv',
    #     'FinancialPhraseBank/Sentences_AllAgree.csv',
    # ],
    'data_path': 'FinancialPhraseBank/Sentences_AllAgree.csv',
    # 'data_path': 'FinancialPhraseBank/forTest.csv',

    # Model hyperparameters
    # 'learning_rates': [2e-6, 2e-5, 2e-4, 2e-3],
    'learning_rates': 2e-5,  # 修改为列表形式，即使只有一个值
    'dropout_rate': 0.3,
    'max_seq_length': 315,
    
    # Training hyperparameters
    # 'batch_size': [8, 16, 32, 64, 128],
    'batch_size': 8,
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
    
    # Visualization settings
    'generate_prediction_samples': True,  # 是否生成前100项预测的提取结果（CSV和PNG）
    
    # Other settings
    'random_seed': 42,
    'model_dir': 'saved_model',
    
    # Output directories
    'output_dir': 'outputs',
    'subdirs': {
        'prediction_results': 'prediction_results',
        'model_analysis': 'model_analysis',
        'benchmark_results': 'benchmark_results'  # 添加benchmark_results到subdirs
    }
}

def ensure_output_dirs():
    """确保所有输出目录存在"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    for subdir in CONFIG['subdirs'].values():
        path = os.path.join(CONFIG['output_dir'], subdir)
        os.makedirs(path, exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

def train_model_with_history(train_dataset, val_dataset, epochs=3, batch_size=16, learning_rate=2e-5, dropout_rate=0.3, visualizer=None):
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
    
    # 如果没有提供可视化器，创建一个新的
    if visualizer is None:
        visualizer = TrainingVisualizer(CONFIG['output_dir'])
    
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
        visualizer.update_metrics(avg_train_loss, train_accuracy, val_accuracy)
        visualizer.plot_training_metrics()
        
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
    
    # 自动检测哪个超参数是列表形式
    param_lists = {
        'loss_function': CONFIG['loss_function'] if isinstance(CONFIG['loss_function'], list) else None,
        'learning_rates': CONFIG['learning_rates'] if isinstance(CONFIG['learning_rates'], list) else None,
        'batch_size': CONFIG['batch_size'] if isinstance(CONFIG['batch_size'], list) else None,
        'data_path': CONFIG['data_path'] if isinstance(CONFIG['data_path'], list) else None
    }
    
    # 找出是列表的超参数
    varying_param = None
    varying_values = None
    for param_name, param_value in param_lists.items():
        if param_value is not None:
            varying_param = param_name
            varying_values = param_value
            break
    
    if varying_param is None:
        print("警告: 没有找到列表形式的超参数，将使用默认值进行单次训练")
        varying_param = 'loss_function'
        varying_values = [CONFIG['loss_function']]
    
    print(f"\n检测到变化的超参数: {varying_param}，值: {varying_values}")
    
    # 创建可视化管理器，用于保存不同参数的训练结果
    viz_manager = TrainingVisualizer(CONFIG['output_dir'])
    
    # 循环使用不同的超参数值进行训练
    for idx, param_value in enumerate(varying_values):
        # 更新当前使用的超参数值
        CONFIG[varying_param] = param_value
        
        # 如果变化的是数据集路径，需要重新加载数据
        if varying_param == 'data_path':
            data_path = CONFIG['data_path']
            if not os.path.exists(data_path):
                print(f"错误: 找不到数据文件 {data_path}")
                continue  # 跳过当前数据集，继续下一个
            
            print(f"\n准备数据集: {data_path}...")
            train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(
                data_path, 
                max_len=CONFIG['max_seq_length']
            )
            
            print(f"训练集大小: {len(train_dataset)}")
            print(f"验证集大小: {len(val_dataset)}")
        else:
            # 如果不是数据集变化，只需要加载一次数据
            if idx == 0:
                data_path = CONFIG['data_path']
                if not os.path.exists(data_path):
                    print(f"错误: 找不到数据文件 {data_path}")
                    return
                
                print("\n准备数据...")
                train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(
                    data_path, 
                    max_len=CONFIG['max_seq_length']
                )
                
                print(f"训练集大小: {len(train_dataset)}")
                print(f"验证集大小: {len(val_dataset)}")
        
        # 获取当前的训练参数
        current_loss_function = CONFIG['loss_function']
        current_learning_rate = CONFIG['learning_rates'] if not isinstance(CONFIG['learning_rates'], list) else CONFIG['learning_rates'][0]
        current_batch_size = CONFIG['batch_size'] if not isinstance(CONFIG['batch_size'], list) else CONFIG['batch_size'][0]
        
        # 为数据集路径创建一个简短的标识符（用于文件名）
        if varying_param == 'data_path':
            # 从路径中提取文件名，去掉扩展名
            dataset_name = os.path.splitext(os.path.basename(param_value))[0]
            display_value = dataset_name  # 使用文件名作为显示值
        else:
            display_value = param_value
        
        print(f"\n开始使用 {varying_param}={display_value} 训练模型 ({idx+1}/{len(varying_values)})...")
        
        # 为每个参数值创建单独的可视化器
        viz = TrainingVisualizer(CONFIG['output_dir'], param_name=varying_param, param_value=display_value)
        
        start_time = time.time()
        
        model = train_model_with_history(
            train_dataset, 
            val_dataset, 
            epochs=CONFIG['epochs'],
            batch_size=current_batch_size,
            learning_rate=current_learning_rate,
            dropout_rate=CONFIG['dropout_rate'],
            visualizer=viz
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{varying_param}={param_value} 的训练完成:")
        print(f"├── 总训练时间: {training_time:.2f}秒")
        print(f"└── 平均每轮时间: {training_time/CONFIG['epochs']:.2f}秒")
        
        # 将当前参数的训练结果添加到可视化管理器
        viz_manager.add_training_result(
            param_name=varying_param,
            param_value=param_value,
            train_losses=viz.train_losses,
            train_accuracies=viz.train_accuracies,
            test_accuracies=viz.test_accuracies
        )
        
        # 在测试集上评估当前模型
        print(f"\n开始在测试集上评估 {varying_param}={param_value} 的模型...")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=current_batch_size)
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
        benchmark_dir = os.path.join(CONFIG['output_dir'], CONFIG['subdirs'].get('benchmark_results', 'benchmark_results'))
        print(f"准备保存分类指标到目录: {benchmark_dir}")
        try:
            # 确保目录存在
            os.makedirs(benchmark_dir, exist_ok=True)
            
            # 如果是数据集变化，使用数据集名称
            if varying_param == 'data_path':
                param_display = dataset_name
            else:
                param_display = str(param_value)
                
            # 保存分类指标
            save_classification_metrics(
                all_labels, 
                all_preds, 
                epoch=f'final_test_{varying_param}_{param_display}', 
                save_dir=benchmark_dir
            )
        except Exception as e:
            print(f"保存分类指标时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 可视化测试集预测结果（根据配置决定是否生成）
        if CONFIG.get('generate_prediction_samples', True):  # 默认为True
            viz.visualize_test_predictions(
                texts=all_texts,
                true_labels=all_labels,
                predicted_labels=all_preds,
                confidences=all_confidences,
                suffix=f'_{varying_param}_{param_value}'
            )
        else:
            print("已跳过生成预测样本可视化（CSV和PNG）")
        
        # 保存当前模型
        if varying_param == 'data_path':
            model_dir = os.path.join(CONFIG['model_dir'], f'{varying_param}_{dataset_name}')
        else:
            model_dir = os.path.join(CONFIG['model_dir'], f'{varying_param}_{param_value}')
        save_model(model, tokenizer, model_dir=model_dir)
        
        print(f"\n开始生成 {varying_param}={param_value} 的测试集预测结果分析...")
        analyze_predictions(model, tokenizer, test_dataset)
    
    # 绘制所有参数值的比较图
    viz_manager.plot_comparison_metrics()

if __name__ == "__main__":
    main()