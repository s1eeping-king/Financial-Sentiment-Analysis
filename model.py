import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=315):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0]  # Take CLS token output
        output = self.drop(pooled_output)
        return self.out(output)

def prepare_data(csv_path, max_len=315, train_split_ratio=0.8):
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
    
    # Split data into train and validation sets
    train_size = int(train_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, tokenizer

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
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.data, 1)
    
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    return sentiment_map[predicted.item()]