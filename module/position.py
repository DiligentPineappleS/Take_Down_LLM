
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
    def __init__(self,dim,max_len=2048,base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len
        # 偶数采样
        inv_freq = 1.0 / (self.base**(torch.arange(0,self.dim,2).float()/self.dim))
        self.register_buffer("inv_freq",inv_freq,persistent=False)

        self.register_buffer("cos_cached",None,persistent=False)
        self.register_buffer("sin_cached",None,persistent=False)
        self.get_rotary_matirix(self.max_len)
    def get_rotary_matirix(self,seq_len):
        seq_cache = torch.arange(seq_len,dtype=torch.get_default_dtype())
        # 外积
        seq_freqs = torch.outer(seq_cache,self.inv_freq)
        # 将 freqs 在最后一个维度（即特征维度）上复制一次,因为inv_freq做了偶数采样
        seq_emb  = torch.cat((seq_freqs,seq_freqs),dim=1)
        self.cos_cached = seq_emb.cos()
        self.sin_cached = seq_emb.sin()
    def forward(self,x,seq_len):
        if seq_len>self.max_len:
            self.get_rotary_matirix(self,seq_len)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype),self.sin_cached[:seq_len].to(dtype=x.dtype))

        
def retate_half(x):
    x1 = x[...,:x.size(-1)/2]
    x2 = x[...,x.size(-1)/2:]
    return torch.cat([-x2,x1],dim=-1)



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
def get_rotary_pos_emb(x,cos, sin, unsqueeze_dim=1):
    # 生成cos/sin时保留序列维度 ；添加维度 [seq_len, 1, 1, dim] → [seq_len, 1, 1, 1, dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    position_embed = (x * cos) + retate_half(x) *sin
    return position_embed
## 拓展一下，提升效率
def get_rotary_pos_emb(x,cos, sin, unsqueeze_dim=1):
    # 生成cos/sin时保留序列维度 ；添加维度 [seq_len, 1, 1, dim] → [seq_len, 1, 1, 1, dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    position_embed = (x * cos) + retate_half(x) *sin
    return position_embed







