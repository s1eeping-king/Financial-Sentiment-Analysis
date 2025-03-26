# Financial Sentiment Analysis Project

This project aims to perform financial sentiment analysis based on the **DistilBERT** model. We train the model using a financial sentence dataset and predict sentiment.

## Project Overview

This project implements a financial text sentiment analysis system that can classify finance-related text as positive, negative, or neutral. The project uses the lightweight DistilBERT model as a foundation and has been optimized for the financial domain.

## Main Features

- DistilBERT-based sentiment analysis model
- Support for multiple loss functions (Cross Entropy, Focal Loss, Label Smoothing, etc.)
- Real-time training visualization
- Early stopping mechanism to prevent overfitting
- Detailed prediction result analysis and visualization
- Model performance evaluation and benchmarking

## Getting the Project

Clone this project:

```bash id=bash0
git clone https://github.com/s1eeping-king/Financial-Sentiment-Analysis.git
```

## Configuration

The CONFIG dictionary in `train.py` can be configured with the following parameters:

```python id=python1
CONFIG = {
    'loss_function': 'cross_entropy',  # Loss function type
    'learning_rate': 2e-5,            # Learning rate
    'batch_size': 128,                # Batch size
    'epochs': 10,                     # Number of training epochs
    'use_early_stopping': False,      # Whether to enable early stopping
    # ... other configuration items
}
```

## Usage

1. Prepare the data:
   - Place the dataset in the `FinancialPhraseBank` directory
   - Supports CSV format, must include text and label columns

2. Train the model:

```bash id=bash2
python train.py
```

1. View results:
   - Training process visualization and performance comparison of different hyperparameter combinations: `outputs/visualization_results/`
   - Analysis of first 100 prediction results: `outputs/visualization_results/test_predictions.png`
   - Performance evaluation report: `outputs/benchmark_results/`
   - Additional visualization analysis:
       - `outputs/model_analysis/`
       - `outputs/prediction_results/`

## Visualization Features

The project provides rich visualization features:

1. Training process visualization:
   - Training loss curve
   - Training accuracy curve
   - Validation accuracy curve

2. Prediction result analysis:
   - Detailed prediction results for the first 100 test samples
   - Includes input text, true label, predicted label, and confidence
   - Incorrect predictions are highlighted

## Performance Optimization

1. Early stopping mechanism:
   - Configurable monitoring metrics (validation accuracy or training loss)
   - Adjustable patience and minimum improvement threshold
   - Automatic saving of the best model

2. Loss function selection:
   - Cross Entropy Loss
   - Focal Loss
   - Label Smoothing
   - Weighted Cross Entropy
   - Dice Loss
