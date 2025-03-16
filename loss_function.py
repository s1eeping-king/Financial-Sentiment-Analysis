import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, 
                             weight=self.weight,
                             reduction=self.reduction)

class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.1, n_classes=3, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * logprobs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, weights=None, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets,
                             weight=self.weights,
                             reduction=self.reduction)

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 将输入转换为概率
        probs = F.softmax(inputs, dim=1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # 计算每个类别的Dice系数
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # 计算Dice损失
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_loss_function(loss_name, **kwargs):

    loss_functions = {
        'cross_entropy': CrossEntropyLoss,
        'focal': FocalLoss,
        'label_smoothing': LabelSmoothingLoss,
        'weighted_cross_entropy': WeightedCrossEntropyLoss,
        'dice': DiceLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"未知的损失函数: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
