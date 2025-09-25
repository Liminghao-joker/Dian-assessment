# Task 3 实现 ViT 的前向传播模型
复现 ViT 前向传播模型
## 代码结构
`my_vit.py`主要类：
- `PatchEmbedding`
```python
Conv2d(k=patch, s=patch) -> Flatten -> LayerNorm(D)
```
- `MHA`
```python
 QKV -> scaled dot-product attention -> 合并头
```
- `EncoderBlock`
	- `x += DropPath(Attn(LN(x)))`
	- `x += DropPath(MLP(LN(x)))`
- `DropPath`
```python
	x = x + self.drop_path(self.attn(self.norm1(x))[0])  
	x = x + self.drop_path(self.mlp(self.norm2(x)))
```
- `ViT`
```python
# 入口函数
def forward_features(self, x):  
    # Patch Embedding  
    x = self.patch_embed(x) # [B, num_patches, embed_dim]  
    # cls_token: [1, 1, embed_dim] -> [B, 1, embed_dim]    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  
    x = torch.cat((cls_token, x), dim=1) # [B, num_patches + 1, embed_dim]  
    x = x + self.pos_embed # [B, num_patches + 1, embed_dim]  
    x = self.pos_dropout(x)  
    for blk in self.blocks:  
        x = blk(x) # [B, num_patches + 1, embed_dim]  
    x = self.norm(x) # [B, num_patches + 1, embed_dim]  
    # extract class token    return x[:, 0]  
  
  
def forward(self, x):  
    # x: [B, C, H, W]  
    x = self.forward_features(x) # [B, embed_dim]  
    return self.head(x) # [B, num_classes]
```
## 形状变化
此处以原论文ViT-Base/16为例

| 点位           | 张量形状             |
| ------------ | ---------------- |
| 输入图像         | (B, 3, 224, 224) |
| Patch 后      | (B, 196, 768)    |
| 拼接 CLS token | (B, 197, 768)    |
| 加 Pos        | (B, 197, 768)    |
| 经过 L 层       | (B, 197, 768)    |
| 取 CLS + Head | (B, K)           |
```python
assert H % patch_size == 0
assert W % patch_size == 0
assert d_models % num_heads == 0
```
## 设计
### Pre-Norm
把`LayerNorm`放在**注意力子层**和**MLP子层**之前，缓解梯度消失
```python
x = x + self.drop_path(self.attn(self.norm1(x))[0])
x = x + self.drop_path(self.mlp(self.norm2(x)))
```
### MLP Head（FFN）
`Linear(D,4D) + GELU + Linear(4D,D)`
### DropPath
- 按样本随机丢弃“残差分支”的输出，并按`1/(1-p)`缩放保留分支；推理时关闭，进行恒等映射
- 作用在每个Encoder的两个**残差连接**
- DropPath率采用**线性递增**，在浅层采用较小的`drop_path_rate`，深层采用较大的`drop_path_rate`
```python
# linear decay of drop path rate  
dpr = torch.linspace(0, drop_path_rate, depth).tolist()
```
### Position Embedding
补充序列位置信息；长度为`N+1`(含`[CLS]`)
`[CLS]`**Token** 作为整个图像特征的输出，送分类头
### Initialization
稳定训练分布
```python
trunc_normal_(std = 0.02) + LN(1,0) # (weight, bias)
```
## 配置表
```python
# Parameter
config = {  
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
}
```
## 自检
运行`my_vit.py`文件的`quick_test()`函数，结果如下图。
`device=cuda, logits_shape=(2, 10), loss=2.4120, pred=[3, 3]`
- 检测维度变换
- 基本训练流程
	- 前向、反向传播
	- 损失计算
	- 参数优化更新
## Reference
>- ViT原始论文 https://arxiv.org/pdf/2010.11929/1000
>-  CSDN Vision Transformer详解
> https://blog.csdn.net/qq_37541097/article/details/118242600
> - 视频
> [使用pytorch搭建Vision Transformer(vit)模型_bilibili](https://www.bilibili.com/video/BV1AL411W7dT/?spm_id_from=333.337.search-card.all.click&vd_source=47dbec3f3db6a86044a31f482a95d4f0)