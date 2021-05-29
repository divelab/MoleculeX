import torch
from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class bce_loss(nn.Module):
    def __init__(self, weights=None, use_gpu=False):
        super(bce_loss, self).__init__()
        self.weights = weights
        self.use_gpu = use_gpu

    def forward(self, outputs, label):
        probs = outputs
        if self.weights is not None:
            fore_weights = torch.tensor(self.weights[0])
            back_weights = torch.tensor(self.weights[1])
            if self.use_gpu:
                fore_weights = fore_weights.to('cuda')
                back_weights = back_weights.to('cuda')
            weights = label * back_weights + (1.0 - label) * fore_weights
        else:
            weights = torch.ones(label.shape)
            if self.use_gpu:
                weights = weights.to('cuda')

        loss = F.binary_cross_entropy(probs, label, weights, reduction='sum') / outputs.shape[0]
        return loss


class masked_bce_loss(nn.Module):
    def __init__(self):
        super(masked_bce_loss, self).__init__()

    def forward(self, outputs, label, mask):
        probs = outputs
        loss_m = F.binary_cross_entropy(probs, label, reduction='none')
        masked_loss_m = torch.sum(loss_m * mask, dim=0) / torch.clamp(torch.sum(mask, dim=0), min=1)
        loss = torch.sum(masked_loss_m)
        return loss


class NTXentLoss_atom(nn.Module):
    def __init__(self, T=0.1):
        super(NTXentLoss_atom, self).__init__()
        self.T = T
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, out, out_mask, labels):
        out = nn.functional.normalize(out, dim=-1)
        out_mask = nn.functional.normalize(out_mask, dim=-1)

        logits = torch.matmul(out_mask, out.permute(0,2,1))
        logits /= self.T

        softmaxs = self.softmax(logits)
        loss = self.criterion(softmaxs.transpose(1,2), labels)

        return loss, logits