import torch
import torch.nn as nn
import math

# dot product attention
def dot_production_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, attn_dropout:float=0.,training:bool=False, mask:torch.Tensor=None):
    # q, k, v: (batch, seq_len, d) or (batch, num_heads, seq_len, d)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]) # (batch, num_heads, seq_len, seq_len)
    if mask is not None:
        # 若 mask 没有head的维度，则扩展一维
        if mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    attn = nn.functional.dropout(attn, p=attn_dropout, training=training)
    out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, d)
    return out, attn

# Drop Path
class DropPath(nn.Module):
    def __init__(self, drop_prob:float=0.,):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x:torch.Tensor):
        # identity mapping when training
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # x:[batch_size, seq_len, embed_dim]
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1) # [batch_size, 1, 1]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # random voice
        random_tensor.floor_() # only 0 or 1

        assert keep_prob > 0, "keep_prob must be greater than 0"
        output = x.div(keep_prob) * random_tensor
        return output

# Linear Projection of Flattened Patches
class PatchEmbedding(nn.Module):
    """
    2D image to Patch Embedding
    [B, C, H, W] -> [B, num_patches, embed_dim]
    return x: [B, num_patches, embed_dim]
    """
    def __init__(self, img_size=224, channels=3, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size # 224
        self.channels = channels # 3
        self.patch_size = patch_size # 16
        self.embed_dim = embed_dim # 768

        # 14*14
        self.num_patches = (img_size // patch_size) * (img_size // patch_size) # 196
        # kernal:[patch_size, patch_size, embed_dim]
        self.conv = nn.Conv2d(in_channels=channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"H, W must be divided by patch_size."
        # [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size] -> [B, embed_dim, num_patches]
        x = self.conv(x) # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2) # [B, embed_dim, num_patches]
        x.transpose_(1, 2) # [B, num_patches, embed_dim]
        return self.norm(x)

# Multi-Head Attention
class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads=12, attn_dropout=0., proj_dropout=0.):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divided by num_heads"
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        # [batch_size, (num_patches + 1), embed_dim]
        B, N, C = x.shape

        # q,k,v: [batch_size, num_patches + 1, 3 * embed_dim]
        qkv = self.qkv(x)
        # q,k,v: [batch_size, num_patches + 1, 3, num_heads, head_dim]
        # q,k,v: [3, batch_size, num_heads, num_patches + 1, head_dim]
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x, attn = dot_production_attention(q, k, v, self.attn_dropout, self.training, mask=None)
        # return x: [batch_size, num_heads, num_patches + 1, head_dim]
        x = x.transpose(1, 2).contiguous().reshape(B, N, C) # [batch_size, num_patches + 1, embed_dim]
        x = self.proj_dropout(self.proj(x))
        return x, attn

# MLP
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:float = 0.):
        # 以 ViT-B/16 为例，d_model:768, d_ff:3072
        # 此处的 seq_len = num_patches + 1
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, attn_dropout:float=0., proj_dropout:float=0., drop_path_rate:float=0., num_heads=12):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim) # before MHA
        self.norm2 = nn.LayerNorm(embed_dim) # after MHA, before MLP
        self.attn = MHA(embed_dim, attn_dropout=attn_dropout, proj_dropout=proj_dropout, num_heads=num_heads)
        self.mlp = MLP(d_model=embed_dim, d_ff=embed_dim*4, dropout=proj_dropout)
        # residual connection and drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        # x:[batch_size, (num_patches + 1), embed_dim]
        x = x + self.drop_path(self.attn(self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x # [batch_size, (num_patches + 1), embed_dim]

# Vision Transformer
class ViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 channels=3,
                 num_heads=12,
                 embed_dim=768,
                 depth=12,
                 dropout=0.,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 drop_path_rate=0.,
                 num_classes=1000,
    ):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, channels=channels, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # [cls] token: [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # position embedding: [1, num_patches + 1, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Dropout before Transformer Encoder
        self.pos_dropout = nn.Dropout(dropout)

        # linear decay of drop path rate
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, attn_dropout=attn_dropout, proj_dropout=proj_dropout, drop_path_rate=dpr[i], num_heads=num_heads)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # MLP head
        self.head = nn.Linear(embed_dim, num_classes)

        # weight init
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_weights)


    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        # cls_token: [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed # [B, num_patches + 1, embed_dim]
        x = self.pos_dropout(x)
        for blk in self.blocks:
            x = blk(x) # [B, num_patches + 1, embed_dim]
        x = self.norm(x) # [B, num_patches + 1, embed_dim]
        # extract class token
        return x[:, 0]


    def forward(self, x):
        # x: [B, C, H, W]
        x = self.forward_features(x) # [B, embed_dim]
        return self.head(x) # [B, num_classes]

def _init_weights(m):
    # m: module
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# simple test
def quick_test():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # simple test
    model = ViT(
        img_size=32,
        patch_size=8,
        channels=3,
        num_heads=4,
        embed_dim=64,
        depth=2,
        dropout=0.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        drop_path_rate=0.0,
        num_classes=10,
    ).to(device)
    model.train()

    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.tensor([1, 2], device=device) % 10

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    opt.step()

    pred = logits.argmax(dim=-1)
    print(f"device={device.type}, logits_shape={tuple(logits.shape)}, loss={loss.item():.4f}, pred={pred.tolist()}")

if __name__ == "__main__":
    quick_test()
    pass
