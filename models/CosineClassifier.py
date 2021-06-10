import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class CosineClassifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048):
        super(CosineClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out