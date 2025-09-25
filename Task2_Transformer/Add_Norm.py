# 实现一个基本的 Add & Norm 模块
import torch
from torch import nn

class AddNorm(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor, sublayer:nn.Module):
        # x: (batch_size, seq_len, d_model)
        # residual connection
        return x + self.dropout(sublayer(self.norm(x)))

# 实现 Feed Forward 模块
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        # x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x