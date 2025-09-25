# 实现 EncoderLayer 和 Encoder
import torch
from torch import nn
from Multi_Head_Attention import MultiHeadAttention
from Add_Norm import AddNorm, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        # x: (batch_size, seq_len, d_model)
        # first residual block
        x = self.add_norm1(x, lambda z: self.mha(z, z, z, mask)[0])
        # second residual block
        x = self.add_norm2(x, self.ffn)
        return x

# 将 EncoderLayer 堆叠成 Encoder
class Encoder(nn.Module):
    def __init__(self, num_layers:int, d_model:int, num_heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    # for test
    def get_layer(self, idx) -> EncoderLayer:
        return self.layers[idx]

if __name__ == "__main__":
    # transformer base
    # random matrix without embedding
    # config_test_2 = {
    #     "batch_size": 2,
    #     "seq_len": 10,
    #     "d_model": 512,
    #     "num_heads": 8,
    #     "d_ff": 2048,
    #     "num_layers": 6,
    #     "dropout": 0.1
    # }
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    print("Input shape:", x.shape)

    mask = torch.ones(batch_size, seq_len, seq_len) # (batch_size, seq_len, seq_len)
    print("Mask shape:", mask.shape)

    encoder = Encoder(num_layers, d_model, num_heads, d_ff)
    out = encoder(x, mask)
    print("Output shape:", out.shape)  # (batch_size, seq_len, d_model)




