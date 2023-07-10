import torch
import torch.nn as nn
class FocalTverskyLoss(nn.Module):

    def __init__(self, epsilon: float = 1e-5, smooth: float = 1., alpha: float = 0.7, gamma: float = 0.75):
        super().__init__()
        self.epsilon = epsilon
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.LongTensor, y_true: torch.LongTensor):
        pt_1 = self.tversky(y_pred, y_true)
        return torch.pow((1 - pt_1), self.gamma)

    def tversky(self, y_pred: torch.LongTensor, y_true: torch.LongTensor):
        y_true_pos = y_true.flatten()
        y_pred_pos = y_pred.flatten()
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)

    def tversky_loss(self, y_pred: torch.LongTensor, y_true: torch.LongTensor):
        return 1 - self.tversky(y_true,y_pred)


