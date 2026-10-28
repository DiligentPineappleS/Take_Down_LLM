
import torch
import torch.nn as nn
import math
# 绝对位置编码
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout_rate,max_len ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # 定义一个向量，保存位置编码的向量矩阵
        pe = torch.zeros(max_len,d_model)

        # 定义位置，从0到最大文本长度的每个位置的索引；一维矩阵max_len---> [max_len,1]
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) 
        # 计算每个位置的频率衰减；指数衰减频率
        div_term = torch.exp(torch.arange(0,d_model,2).float()* (-math.log(10000)/d_model))
        # 填充正弦和余弦值
        # 偶数维度：sin(pos / 10000^(2i/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        # 奇数维度：cos(pos / 10000^(2i/d_model))
        pe[:,1::2] = torch.cos(position * div_term)
        # 扩展维度以适配batch输入：[max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0,1)
        # 注册为Buffer（非可训练参数，但会随模型保存/加载）
        self.register_buffer("pe",pe)
    def forward(self,x):
        x = x + self.pe[:x.size(1),:]
        return self.dropout(x)



# 相对位置编码
class RelativePositionalEncoding(nn.Module):
    def __init__(self,num_heads,max_relative_distance ):
        super().__init__()
        self.num_heads = num_heads
        self.max_ralative_distance = max_relative_distance
        self.embedding = nn.Embedding(num_embeddings=2*max_relative_distance+1,embedding_dim=num_heads)
        nn.init.trunc_normal_(self.embedding.weight,std =0.02)

    def forward(self,seq_len):
        range_vec = torch.range(seq_len)
        relative_pos = range_vec[None,:] - range_vec[:,None]
        relative_pos = torch.clamp(relative_pos,min= -self.max_ralative_distance,max = self.max_ralative_distance)
        relative_pos += self.max_ralative_distance
        bias = self.embedding(relative_pos.to(self.embedding.weight.device))
        bias = bias.permute(2,0,1).contiguous()
        return bias


from transformers import AutoModelForCausalLM, AutoTokenizer
# ​​旋转位置编码
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim=None, max_len=2048, base=10000, hidden_dim=None):
        super().__init__()
        if dim is None:
            dim = hidden_dim
        if dim is None:
            raise ValueError("RotaryPositionalEncoding 需要 dim 或 hidden_dim")
        self.dim = dim
        self.base = base
        self.max_len = 0
        # 构造时传入的宽度可能是 hidden_dim，真正旋转的是 x 的最后一维。
        # 宽度为偶数时先建缓存；奇数则等 forward 按 x 的 rope 维重建。
        if dim % 2 == 0:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.get_rotary_matirix(max_len)
        else:
            self.register_buffer("inv_freq", torch.empty(0), persistent=False)
            self.register_buffer("cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def get_rotary_matirix(self, seq_len):
        seq_cache = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # 外积得到 [seq, dim/2]，再复制成 [seq, dim]，与前后对半旋转对齐
        seq_freqs = torch.outer(seq_cache, self.inv_freq)
        seq_emb = torch.cat((seq_freqs, seq_freqs), dim=-1)
        self.register_buffer("cos_cached", seq_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", seq_emb.sin(), persistent=False)
        self.max_len = seq_len

    def forward(self, x, seq_len=None):
        rope_dim = x.size(-1)
        cur_len = x.size(-2)
        if rope_dim % 2 != 0:
            raise ValueError("RoPE 的最后一维必须是偶数")
        # 频率按正在旋转的向量宽度计算，而不是构造时传入的 hidden_dim
        if self.inv_freq.numel() * 2 != rope_dim or self.inv_freq.device != x.device:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.dim = rope_dim
            self.max_len = 0
        cache_ready = (
            self.max_len >= cur_len
            and self.cos_cached.numel() > 0
            and self.cos_cached.size(-1) == rope_dim
            and self.cos_cached.device == x.device
        )
        if not cache_ready:
            self.get_rotary_matirix(max(cur_len, self.max_len))
        # 序列长度以 x 的倒数第二维为准，返回 (cos, sin)，形状 [seq, rope_dim]
        cos = self.cos_cached[:cur_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:cur_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


def retate_half(x):
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)



"""
# 复数旋转公式：rot = (a*cosθ - b* sinθ) + i *(a*sinθ + b *cosθ) ;
# 其中：a = q[...,:dim/2];b = q[...,dim/2:] ;
# 实部和虚部 -->组合为实数向量：[a*cosθ - b* sinθ, a*sinθ + b *cosθ] ，
# 进而采用矩阵乘法转化：
[cosθ -sinθ [a
 sinθ  cosθ] b] 
# rot = [a*cosθ-b*sinθ, a*sinθ+b *cosθ]
      = [acosθ, bcosθ]+[−bsinθ, asinθ]
      = [a, b]⋅cosθ+[−b, a]⋅sinθ

"""
def get_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    # cos/sin: [seq, rope_dim]，或已经带 batch 的 [batch, seq, rope_dim]
    # x 的最后两维同样是 [seq, rope_dim]
    if cos.size(-1) != x.size(-1) or cos.size(-2) != x.size(-2):
        raise ValueError(
            f"cos/sin 的序列长度和 rope 维必须与 x 的最后两维一致，cos {tuple(cos.shape)}，x {tuple(x.shape)}"
        )
    if cos.dim() == x.dim() - 1 and cos.dim() >= 3 and cos.size(0) == x.size(0):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    else:
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
    return (x * cos) + retate_half(x) * sin







