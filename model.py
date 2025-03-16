import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast  # 添加自动混合精度支持

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=315):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 预计算所有样本的编码，避免训练时重复计算
        print("预计算数据集编码...")
        self.encodings = self.tokenizer(
            list(map(str, self.texts)),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)
        
        # 使用正交初始化提高收敛速度
        nn.init.orthogonal_(self.out.weight)
        
    def forward(self, input_ids, attention_mask):
        with autocast():  # 使用自动混合精度
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs[0][:, 0]  # Take CLS token output
            output = self.drop(pooled_output)
            return self.out(output)

def prepare_data(csv_path, max_len=315, train_ratio=0.8, val_ratio=0.1):
    # Read and preprocess data
    df = pd.read_csv(csv_path)
    
    # Convert sentiment labels to numerical values
    sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    df['sentiment'] = df['sentiment'].map(sentiment_map)
    
    texts = df['text'].values
    labels = df['sentiment'].values
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create dataset
    dataset = SentimentDataset(texts, labels, tokenizer, max_len=max_len)
    
    # Split data into train, validation and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def predict_sentiment(text, model, tokenizer):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    with torch.no_grad(), autocast():  # 使用自动混合精度
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.data, 1)
    
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    return sentiment_map[predicted.item()]