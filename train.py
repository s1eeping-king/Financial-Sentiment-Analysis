import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from model import SentimentClassifier, prepare_data, predict_sentiment

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
    'model_dir': 'saved_model'
}

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
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 训练模型
    print("\n开始训练模型...")

    # 开始基准测试
    start_time = time.time()
    model, train_losses, val_accuracies = train_model_with_history(
        train_dataset, 
        val_dataset, 
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        dropout_rate=CONFIG['dropout_rate']
    )
    end_time = time.time()
    
    # 输出训练时间
    print(f"训练模型所需时间: {end_time - start_time:.2f}秒")
    
    # 保存模型
    save_model(model, tokenizer)
    
    # 可视化训练过程
    plot_training_history(train_losses, val_accuracies)
    
    # 测试模型
    test_model(model, tokenizer)

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
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'平均训练损失: {avg_train_loss:.4f}')
        print(f'验证准确率: {val_accuracy:.2f}%\n')
    
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
    plt.savefig('training_history.png')
    print("训练历史图表已保存为 'training_history.png'")
    plt.close()

def test_model(model, tokenizer):
    """测试模型在一些示例文本上的表现"""
    print("\n测试模型在示例文本上的表现:")
    
    test_texts = [
        "I love this product, it works perfectly!",
        "This is the worst experience I've ever had.",
        "The product arrived on time and works as expected.",
        "Are you sure this is a good idea?"
    ]
    
    for text in test_texts:
        sentiment = predict_sentiment(text, model, tokenizer)
        print(f"\n文本: {text}")
        print(f"预测情感: {sentiment}")

if __name__ == "__main__":
    main()