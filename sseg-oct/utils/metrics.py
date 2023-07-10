import torch
from typing import Dict, Callable
from sklearn.metrics import jaccard_score
class Metric:
    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics
        self.values = dict(zip(metrics.keys(), [0 for _ in range(len(metrics))]))
        self.n = 0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        for metric, func in self.metrics.items():
            self.values[metric] += func(y_pred, y_true)
        self.n += 1

    def __repr__(self):
        return ', '.join([f'{metric}={round(value/self.n, 3)}' for metric, value in self.values.items()])

    def __getitem__(self, metric):
        return self.values[metric]/self.n



def precision(y_pred: torch.Tensor, y_true: torch.Tensor):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    try:
        return tp/(tp+fp)
    except ZeroDivisionError:
        return 0

def recall(y_pred: torch.Tensor, y_true: torch.Tensor):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    try:
        return tp/(tp+fn)
    except ZeroDivisionError:
        return 0

def fscore(y_pred: torch.Tensor, y_true: torch.Tensor):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    try:
        return (2*tp)/(2*tp+fp+fn)
    except ZeroDivisionError:
        return 0


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    acc = ((y_true == y_pred).sum() / len(y_true)).item()
    return acc


def iou(y_pred: torch.Tensor, y_true: torch.Tensor):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    try:
        return tp/(tp+fn+fp)
    except ZeroDivisionError:
        return 0

