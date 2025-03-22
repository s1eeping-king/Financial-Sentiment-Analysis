import numpy as np

class EarlyStopping:
    """早停机制
    
    当验证集上的性能指标在连续patience个epoch内没有改善时，
    提前终止训练以防止过拟合
    
    Args:
        patience (int): 等待改善的轮数
        mode (str): 'min' 用于监控最小化指标(如损失), 'max' 用于监控最大化指标(如准确率)
        min_delta (float): 被认为是改进的最小变化量
        verbose (bool): 是否打印早停信息
    """
    def __init__(self, patience=3, mode='min', min_delta=0, verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # 根据mode设置比较运算
        if mode == 'min':
            self._is_better = lambda score, best: score < (best - min_delta)
        elif mode == 'max':
            self._is_better = lambda score, best: score > (best + min_delta)
        else:
            raise ValueError(f"mode {mode} is unknown")
    
    def __call__(self, epoch, score):
        """
        Args:
            epoch (int): 当前epoch
            score (float): 当前需要监控的指标值
            
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self._is_better(score, self.best_score):
            if self.verbose:
                improvement = abs(score - self.best_score)
                print(f'性能改善: {improvement:.6f}')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'触发早停! 最佳性能出现在epoch {self.best_epoch + 1}')
                return True
        
        return False
    
    def get_best_score(self):
        """返回监控指标的最佳值"""
        return self.best_score
    
    def get_best_epoch(self):
        """返回取得最佳性能的epoch"""
        return self.best_epoch 