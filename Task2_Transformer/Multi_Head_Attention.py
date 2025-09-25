# 实现一个基本的Multi-Head Attention层
import torch
from torch import nn
from Self_Attention import dot_production_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # make sure d_model can be divided by num_heads

        self.d_k = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # linear projections
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = q.view(q.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        key = k.view(k.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)

        # attention score
        # x: (batch_size, num_heads, seq_len, d_k)
        # attn: (batch_size, num_heads, seq_len, seq_len)
        x, attn = dot_production_attention(query, key, value, mask)

        # concat heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.Wo(x)
        return x, attn

if __name__ == "__main__":
    # simple test
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]).unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, seq_len)
    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    out, attn = multi_head_attention(x, x, x, mask)
    print("Output shape:", out.shape)  # (batch_size, seq_len, d_model)
    print("Attention shape:", attn.shape)  # (batch_size, num_heads, seq_len, seq_len)
    print("Output:", out)
    print("Attention:", attn)




