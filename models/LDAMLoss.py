import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, cls_count, max_m=0.5, weight=None, s=30, device='cpu'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / cls_count.float().sqrt().sqrt()
        self.m_list = m_list * (max_m / m_list.max())
        self.m_list = self.m_list.to(device)
        assert s > 0
        self.s = s
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float().to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
