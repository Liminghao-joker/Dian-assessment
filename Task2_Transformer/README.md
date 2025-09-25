# Task 2 Transformer Encoder
实现 Self-Attention, Multi-Head Attention, Add&Norm, Feed Forward，并堆叠为 Encoder，并用随机矩阵输入进行测试。
## 项目结构
- `Self_Attention.py`
	- Scaled Dot-Product Attention 
	- Self-Attention Block  
- `Multi_Head_Attention.py`
	- 多头拆分、拼接与输出线性层，处理多头注意力
- `Add_Norm.py`
	- 包含AddNorm 和 FFN 的实现
	- 残差 + LayerNorm + Dropout
- `Encoder.py`
	- `EncoderLayer` 与堆叠的 `Encoder`
	- 测试参数配置二，检查输入、输出维度
- `test.py`
	- 测试参数配置一，打印第一层各过程张量

## 测试
测试使用参数
```python
# 打印 encoder 第一层各过程张量
config_test_1 = {  
    "batch_size": 2,  
    "seq_len": 4,  
    "d_model": 8,  
    "num_heads": 2,  
    "d_ff": 16,  
    "num_layers": 2
    "dropout": 0.1  
}

# transformer base
# 检查输入、输出的维度变化
config_test_2 = {  
    "batch_size": 2,  
    "seq_len": 10,  
    "d_model": 512,  
    "num_heads": 8,  
    "d_ff": 2048,  
    "num_layers": 6,  
    "dropout": 0.1  
}
```
在 *Pycharm* 的*Python* 控制台检测各张量的维度变化，结果符合原预期。

### 结果
**输入随机矩阵**
**Padding mask**

**注意力权重矩阵**
**输出结果**
### 常见问题及解决
1. 维度不匹配
	解决分头后的维度匹配问题，需注意`assert d_model % num_heads == 0`
2. Padding Mask 的形状 广播机制
3. Add&Norm 中归一化norm
### 学习
1. 为什么要采用”多头注意力机制“？
2. Pre-LN
