
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from layernorm import RMSNorm
from position import get_rotary_pos_emb,RotaryPositionalEncoding
# 自注意力机制(缩放点积注意力:自注意力基础上加了dropout和掩码填充)
 
class SelfAttention(nn.Module):
    def __init__(self,d_model,dropout_rate=None):
        super().__init__()
        self.d_model = d_model
        self.Key = nn.Linear(self.d_model,self.d_model)
        self.Value = nn.Linear(self.d_model,self.d_model)
        self.Query = nn.Linear(self.d_model,self.d_model)
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,x,attention_mask=None):
        Key = self.Key(x) # [batch_size, seq_len, d_model]
        Value = self.Value(x) # [batch_size, seq_len, d_model]
        Query = self.Query(x) # [batch_size, seq_len, d_model]

        attention_value= torch.matmul(Query,Key.transpose(-1,-2)) # 交换key的最后两个维度，使其变换为[batch_size, d_model, seq_len]；输出：[batch_size, seq_len, seq_len]
        attention_score = attention_value/math.sqrt(self.d_model) 

        if attention_mask:
            # 掩码矩阵attention_mask和注意力分数矩阵的形状是完全一致的
            attention_score = attention_score.masked_fill(attention_mask==0,float("-1e20")) # 将 attention_score 中所有与 attention_mask == 0 对应的位置替换为一个极小的负值（-1e20，为了确保在 softmax中这些位置的权重趋近于 0）
        attention_weight = F.softmax(attention_score,dim=-1) 
        if self.dropout_rate:
            attention_weight = self.dropout(attention_weight) 
        attention_output = torch.matmul(attention_weight,Value)

        return attention_output

# 多头注意力机制
## 原理：通过将输入分成多个独立的头，并行计算注意力，每个头学习不同的上下文关系
class MultiheadAttention(nn.Module):
    def __init__(self, d_model,num_heads,dropout_rate = None):
        super(MultiheadAttention).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0, "d_model 必须被num_heads整除"
        self.dim = d_model//num_heads 
        self.Key = nn.Linear(self.d_model,self.d_model)
        self.Value = nn.Linear(self.d_model,self.d_model)
        self.Query = nn.Linear(self.d_model,self.d_model)
        self.Output = nn.Linear(d_model, d_model)  # 输出层
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,query,key,value,attention_mask=None):
        batch_size,seq_len,hidden_dim = query.shape
        Key = self.Key(key) # [batch_size, seq_len, d_model]
        Value = self.Value(value) # [batch_size, seq_len, d_model]
        Query = self.Query(query) # [batch_size, seq_len, d_model]
        # 将key,value,query拆分成多个头，从 [batch_size, seq_len, d_model] 变换为[batch_size, seq_len, num_heads, head_dim],将最后一个维度d_model拆分为num_heads* head_dim
        # 将self.num_heads的维度提前，交换self.num_heads和seq_len位置，如此可以实现并行计算，整个变换没有修改矩阵的元素，仅仅修改矩阵的形状，理解这一步就理解的MHA了
        # 通俗点说就是有一个篇加密的文章有batch_size句话，每一句话有seq_len个字，每个字有self.d_model个加密数字，
        # 那我现在解密这篇文章，假设我有self.num_heads个员工，我要分配给他们解密，他们每个人在每句话中就负责解密self.dim加密数字，截止此处完成了分头
        Key = Key.view(batch_size,-1,self.num_heads,self.dim) #[batch_size, seq_len, num_heads, head_dim]
        Value = Value.view(batch_size,-1,self.num_heads,self.dim)
        Query = Query.view(batch_size,-1,self.num_heads,self.dim)
        # 变换key,value,query的维度，[batch_size, seq_len, num_heads, head_dim] 变换为[batch_size, num_heads，seq_len, head_dim]
        # 从原来的逻辑：文章句数-句子字数-员工-分配解码的数字 变换为文章句数-员工-句子字数-分配解码
        # 句子维度根据员工人数进行划分，员工的维度根据句子的进行划分；如此可以保证每个员工的独立性；
        Key = Key.transpose(1,2) #[batch_size, num_heads, seq_len, head_dim]
        Value = Value.transpose(1,2) 
        Query = Query.transpose(1,2)

        # 计算每个词和其他词的相似度：每个员工根据编码的来找到数字和数字之间的关系；
        attention_value= torch.matmul(Query,Key.transpose(-1,-2)) # 交换key的最后两个维度，使其变换为[batch_size, d_model, seq_len]；输出：[batch_size, seq_len, seq_len]
        # 对相似度进行缩放，点积的值可能会很大，因为我们计算点积的维度是self.dim，所以采用sqrt(self.dim)进行缩放
        # 这里选择sqrt(self.dim)，因为点积的方差为self.dim，标准差是sqrt(self.dim)
        attention_score = attention_value/math.sqrt(self.dim)  # [batch, num_heads, seq_len, seq_len]

        if attention_mask:
            # 掩码矩阵attention_mask和注意力分数矩阵的形状是完全一致的
            attention_score = attention_score.masked_fill(attention_mask==0,float("-1e20")) # 将 attention_score 中所有与 attention_mask == 0 对应的位置替换为一个极小的负值（-1e20，为了确保在 softmax中这些位置的权重趋近于 0）
        # 计算注意力权重，采用softmax将计算的每个词和其他词相似度（注意力分数）转化为注意力概率；
        attention_weight = F.softmax(attention_score,dim=-1)  # [batch, num_heads, seq_len, seq_len]
        if self.dropout_rate:
            attention_weight = self.dropout(attention_weight)
        # 计算注意力输出，将注意力权重映射到每个词上，建立起qury和value的注意力关系
        attention_output = torch.matmul(attention_weight,Value)  # [batch, num_heads, seq_len, head_dim]
        # 合并每个头的输出，保持和输入维度一致
        # 1. 维度变换[batch, num_heads, seq_len, head_dim] -->[batch, seq_len, num_heads, head_dim] ;
        # 2. 保证将张量放入连续存储；
        # 3. 张量转换 [batch, seq_len, num_heads, head_dim] --->[batch_size, seq_len, d_model];合并num_heads和head_dim
        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model) 
        # 输出；这一步主要是因为在完成向量拼接后，向量维度是和输入维度一致，但相对表达能力不足；为了能够更好将每个头的信息进行融合，采用线性层将所有头的不同特征进行统一表示
        attention_output = self.Output(attention_output)
 
        return attention_output

# 多查询注意力
## 原理：基于多头注意力，采用共享K，V矩阵的方式来计算     
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model,num_heads,dropout_rate = None):
        super(MultiQueryAttention).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0, "d_model 必须被num_heads整除"
        self.dim = d_model//num_heads 
        # 此处维度进行了转换，将d_model-->dim
        self.Key = nn.Linear(self.d_model,self.dim)
        self.Value = nn.Linear(self.d_model,self.dim)
        self.Query = nn.Linear(self.d_model,self.d_model)
        self.Output = nn.Linear(d_model, d_model)  # 输出层
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,query,key,value,attention_mask=None):
        batch_size,seq_len,hidden_dim = query.shape
        Key = self.Key(key) # [batch_size, seq_len, dim]
        Value = self.Value(value) # [batch_size, seq_len, dim]
        Query = self.Query(query) # [batch_size, seq_len, d_model]  
        # 拆分为多个头
        Query =  Query.view(batch_size,-1,self.num_heads,self.dim).transpose(1,2) #[batch_size, seq_len, d_model] --> [batch_size, seq_len, num_heads,dim] ---> [batch_size,num_heads, seq_len,dim]
        # 增加一个维度，即默认有一个头
        Key = Key.unsqueeze(1) # [batch_size,1, seq_len,dim]
        Value = Value.unsqueeze(1) # [batch_size,1, seq_len,dim]

        # 计算相似度，注意力分数
        attention_score = torch.matmul(Query,Key.transpose(-1,-2))
        attention_score = attention_score/math.sqrt(self.dim)  # [batch_size,num_heads, seq_len,seq_len]

        if attention_mask:
            attention_score = attention_score.masked_fill(attention_mask==0,float("-inf"))
        
        attention_weight = F.softmax(attention_score)  # [batch_size,num_heads, seq_len,seq_len]

        attention_output = torch.matmul(attention_weight,Value)  # [batch_size,num_heads, seq_len,head_dim]
        if self.dropout_rate:
            attention_output = self.dropout(attention_output)


        attention_output = attention_output.transpose(1,2).contiguous().view(-1,-1,self.d_model)

        attention_output = self.Output(attention_output)

        return attention_output

# 分组查询注意力
class GroupAttention(nn.Module):
    def __init__(self, d_model,num_heads,num_groups,dropout_rate = None):
        super(MultiQueryAttention).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0, "d_model 必须被num_heads整除"
        assert num_heads%num_groups==0, "num_heads 必须被num_groups整除"
        self.dim = d_model//num_heads  
        self.num_groups = num_groups

        self.q_num_heads_per_group = num_heads//num_groups 

        # 此处维度进行了转换，将d_model-->dim
        self.Key = nn.Linear(self.d_model,self.dim *self.num_groups)
        self.Value = nn.Linear(self.d_model,self.dim *self.num_groups)
        self.Query = nn.Linear(self.d_model,self.d_model)
        self.Output = nn.Linear(d_model, d_model)  # 输出层
        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self,query,key,value,attention_mask=None):
        batch_size,seq_len,hidden_dim = query.shape

        Key = self.Key(key) # [batch_size, seq_len, self.dim *self.num_groups]
        Value = self.Value(value) # [batch_size, seq_len, self.dim *self.num_groups]
        Query = self.Query(query) # [batch_size, seq_len, d_model] 

        # Query分头
        Query = Query.view(batch_size,-1,self.num_heads,self.dim).transpose(1,2).contiguous() # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_heads,dim]  --> [batch_size, num_heads,seq_len, dim] 

        # Key,Value 分组
        Key = Key.view(batch_size,-1,self.num_groups,self.dim) # [batch_size, seq_len, self.dim *self.num_groups] -->[batch_size, seq_len, self.num_groups,self.dim ]
        Value = Value.view(batch_size,-1,self.num_groups,self.dim)# [batch_size, seq_len, self.dim *self.num_groups] -->[batch_size, seq_len, self.num_groups,self.dim ]
        # Key,Value 维度变换
        Key = Key.transpose(1,2) # [batch_size, seq_len, self.num_groups,self.dim ] -->[batch_size, self.num_groups, seq_len, self.dim ]
        Value = Value.transpose(1,2) # [batch_size, seq_len, self.num_groups,self.dim ] -->[batch_size, self.num_groups, seq_len, self.dim ]
        # Key,Value 增加维度，计算头个数为1
        Key = Key.repeat_interleave(self.q_num_heads_per_group,dim=1) # [batch_size, self.num_groups, seq_len, self.dim ] ---> [batch_size, num_heads, seq_len, self.dim ]
        Value = Value.repeat_interleave(self.q_num_heads_per_group,dim=1) # [batch_size, self.num_groups, seq_len, self.dim ] ---> [batch_size, num_heads, seq_len, self.dim ]
        # 计算相似度
        ## [batch_size, num_head,seq_len, dim] * [batch_size,num_head,dim,seq_len] = [batch_size, num_head, seq_len, seq_len ]
        attention_score = torch.matmul(Query,Key.transpose(-1,-2))/math.sqrt(self.dim)
        # 掩码
        if attention_mask :
            attention_score = attention_score.masked_fill(attention_mask==0,float("-inf"))
        
        # 计算权重
        attention_weight = F.softmax(attention_score,dim=-1) # [batch_size, num_head, seq_len, seq_len ]
        # Droupout 
        if self.dropout_rate:
            attention_weight = self.dropout(attention_weight) 
        # 计算注意力输出
        ## [batch_size, num_head, seq_len, seq_len ] * [batch_size,num_head,seq_len，dim] = [batch_size, num_head, seq_len, dim ]
        attention_output = torch.matmul(attention_weight,Value) 

        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        attention_output = self.Output(attention_output)
        return attention_output


# MLA 
## 下面的代码MLA是完全是为了完善代码体系手搓的，时间比较赶，在位置编码部分简化部分代码，不影响MLA的主要逻辑     
class MultiLatentAttention(nn.Module):
    def __init__(self, hidden_dim,max_length,num_heads,q_lora_rank,q_head_dim,kv_lora_rank,q_rope_dim,v_hidden_dim,num_kv_head,eps = 1e-20,dropout_rate = None):
        super(MultiQueryAttention).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim%num_heads==0, "d_model 必须被num_heads整除"
        assert num_heads%num_kv_head==0, "num_heads 必须被num_groups整除"
        self.num_kv_head = num_kv_head
        self.num_groups = hidden_dim//self.num_kv_head
        self.max_length = max_length
        
        self.eps = eps
        self.dropout_rate = dropout_rate
        
        self.q_lora_rank = q_lora_rank
        self.q_head_dim = q_head_dim
        self.q_rope_dim = q_rope_dim
        self.q_nrope_dim = q_head_dim - q_rope_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_hidden_dim = v_hidden_dim
        self.q_lora_a_net = nn.Linear(self.hidden_dim,self.q_lora_rank,bias=True)
        self.q_lora_rmsnorm = RMSNorm(self.q_lora_rank,self.eps)
        self.q_lora_b_net = nn.Linear(self.q_lora_rank,self.num_heads*self.q_head_dim)

        self.kv_lora_a_net = nn.Linear(self.hidden_dim,self.kv_lora_rank+self.q_rope_dim,bias=True)
        self.kv_lora_rmsnorm = RMSNorm(self.kv_lora_rank,self.eps)
        self.kv_lora_b_net = nn.Linear(self.kv_lora_rank,self.num_kv_head*(self.q_nrope_dim+self.v_hidden_dim))

        self.output_net = nn.Linear(self.num_heads*self.v_hidden_dim,self.hidden_dim,bias=True)
        self.emb_positon = RotaryPositionalEncoding(self.hidden_dim,max_len=max_length,base=10000)
        if dropout_rate:
            self.dropout_net = nn.Dropout(dropout_rate)


    def forward(self,hidden_states:torch.Tensor,attention_mask):
        batch_size,seq_len = hidden_states.shape
        q_lora_a = self.q_lora_a_net(hidden_states)
        q_lora_norm = self.q_lora_rmsnorm(q_lora_a)
        q_lora_b = self.q_lora_a_net(q_lora_norm)
        q_lora_b = q_lora_b.view(batch_size,seq_len,-1, self.q_head_dim).transpose(1,2)

        q_nrope,q_rope = torch.split(q_lora_b,[self.q_nrope_dim,self.q_rope_dim],dim=-1)

        k_lora_a = self.kv_lora_a_net(hidden_states)

        k_lora_a,kv_rope = torch.split(k_lora_a,[self.kv_lora_rank,self.q_rope_dim],dim=-1)
        k_lora_norm = self.kv_lora_rmsnorm(k_lora_a)
        kv_lora_b = self.kv_lora_b_net(k_lora_norm)
        kv_lora_b = kv_lora_b.view(batch_size,seq_len,-1,self.q_nrope_dim+self.v_hidden_dim).transpose(1,2)
        k_nrope,value_states = torch.split(kv_lora_b,[self.q_nrope_dim,self.v_hidden_dim],dim=-1)
        
        sins, cos = self.emb_positon(kv_rope,self.max_length)
        kv_rope = kv_rope.view(batch_size,1,seq_len,self.q_rope_dim)
        kv_rope_emb = get_rotary_pos_emb(kv_rope,sins, cos)
        q_rope_emb = get_rotary_pos_emb(q_rope,sins, cos)
        
        kv_rope_emb = kv_rope_emb.expand(batch_size,seq_len,self.num_kv_head,-1)
        key_states = torch.cat([kv_rope_emb,k_nrope])
        key_states = key_states.unsqueeze(2).expand(batch_size,self.num_kv_head,self.num_groups,seq_len,-1).reshape(batch_size,self.num_heads,seq_len,-1)
        value_states = value_states.unsqueeze(2).expand(batch_size,self.num_kv_head,self.num_groups,seq_len,-1).reshape(batch_size,self.num_heads,seq_len,-1)

        query_states = torch.cat([q_rope_emb,k_nrope])

        # q_rope:[batch_size,num_heads,seq_len,q_head_dim] *[batch_size,num_heads,q_head_dim,seq_len] ---> [batch_size,num_heads,seq_len,seq_len] 
        attention_score = torch.matmul(query_states,key_states.transpose(-1,-2))/torch.sqrt(self.q_head_dim)
        if attention_mask:
            attention_score = attention_score.masked_fill(attention_mask==0,float("-inf"))
        attention_weight = F.softmax(attention_score,dim=-1)
        if self.dropout_rate:
            attention_weight = self.dropout_net(attention_weight)
        # [batch_size,num_heads,seq_len,seq_len] *[batch_size,num_heads,seq_len,v_head_dim] ---> [batch_size,num_heads,seq_len,v_head_dim] 
        attention_output = torch.matmul(attention_weight,value_states)
        
        attention_output = attention_output.transpose(1,2).contiguous().reshape(batch_size,seq_len,-1)# batch_size,seq_len,self.num_heads*self.v_hidden_dim
        output = self.output_net(attention_output)# batch_size,seq_len,hidden_dim
        return output,attention_weight
        

# 注释详解版本MLA 
## 下面代码来自model文件夹中的deepseek3_model.py,先做了deepseekv3的实现，所以deepseek3_model.py中的注释比较详细
## 上面的代码MLA时间比较赶，注释比较少，在位置编码部分简化部分代码，不影响MLA的主要逻辑     
class MLA(nn.Module):
    def __init__(self,hidden_dim, query_head_dim,value_head_dim,num_kv_heads,num_heads,max_len,q_lora_rank,kv_lora_rank,
                 qk_rope_head_dim,qk_nrope_head_dim,keyvalue_head_dim,attention_bias,scaling_factor,is_rope_interleave,mscale_dim,dropout_rate=None,eps=1e-6,base=10000):
        super(MLA,self).__init__()
        assert num_heads%num_kv_heads == 0 ,"num_heads must be num_groups*num_kv_heads,num_kv_heads is int"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads//self.num_kv_heads
        self.query_head_dim = query_head_dim
        self.value_head_dim = value_head_dim
        self.keyvalue_head_dim = keyvalue_head_dim
        self.is_rope_interleave = is_rope_interleave

        # Latent层
        ## Query lora低秩分解维度
        self.q_lora_rank = q_lora_rank
        ## Query 旋转编码维度
        self.qk_rope_head_dim = qk_rope_head_dim
        ## Query 非旋转编码维度
        self.qk_nrope_head_dim = qk_nrope_head_dim ##self.qk_nrope_head_dim = qk_nrope_head_dim ==self.query_head_dim - self.qk_rope_head_dim
        # key/value  lora低秩分解维度
        self.kv_lora_rank = kv_lora_rank

        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.base = base
        self.attention_bias = attention_bias
        self.eps = eps
        # query Latent
        self.query_lora_a_proj = nn.Linear(self.hidden_dim,self.q_lora_rank,bias=self.attention_bias )
        self.query_lora_layernorm = RMSNorm(self.q_lora_rank,self.eps)
        self.query_lora_b_proj = nn.Linear(self.q_lora_rank,self.num_heads*self.query_head_dim ,bias=False)
        # key/value  Latent
        ## kv_a_layernorm: (b,s,r_kv) -> (b,s,r_kv)
        ## kv_b_proj: (b,s,r_kv) -> (b,s,n_head*(d_nope+d_v))
        ## view+transpose: (b,s,n_head,d_nope+d_v) -> (b,n_head,s,d_nope+d_v)
        self.key_value_lora_a_proj = nn.Linear(self.hidden_dim,self.kv_lora_rank+self.qk_rope_head_dim,bias = self.attention_bias)
        self.key_value_lora_layernorm = RMSNorm(self.kv_lora_rank,self.eps)
        ## 此处的维度为什么是self.num_heads *(self.qk_nrope_head_dim + self.value_head_dim)？
        self.key_value_lora_b_proj = nn.Linear(self.kv_lora_rank,self.num_kv_heads *(self.qk_nrope_head_dim + self.value_head_dim),bias=False)
        # 输出
        self.output_proj = nn.Linear(self.num_heads* self.value_head_dim,self.hidden_dim,bias=self.attention_bias)
        if dropout_rate:
            self.atten_dropout = nn.Dropout(self.dropout_rate)
        # RoPE位置编码
        self.rotaryemb = RotaryPositionalEncoding(hidden_dim=self.hidden_dim,max_len=self.max_len,base=self.base)
        # attention动态缩放因子
        self.scaling_factor = scaling_factor
        self.mscale_dim = mscale_dim


    def forward(self,hidden_states:torch.Tensor,attention_mask):
        batch_size,seq_len,hidden_dim = hidden_states.shape
        # query将高维特征压缩为低秩特征， [batch_zise,seq_len,hidden_dim]---->[batch_zise,seq_len,q_lora_rank]
        Query_lora_a_states = self.query_lora_a_proj(hidden_states)  
        # query低秩特征归一化，主要原因是低秩特征的波动较大，使用归一化降低梯度爆炸/消失的风险：[batch_zise,seq_len,q_lora_rank]---->[batch_zise,seq_len,q_lora_rank]
        Query_loranorm_states = self.query_lora_layernorm(Query_lora_a_states) 
        # 多头注意力特征构建 [batch_zise,seq_len,q_lora_rank]----> [batch_zise,seq_len,num_head*query_head_dim]
        Query_lora_b_states = self.query_lora_b_proj(Query_loranorm_states)  
        #矩阵变换，多头矩阵； [batch_zise,seq_len,num_head*query_head_dim] ----> [batch_zise,seq_len,num_head,query_head_dim]
        Query_states =  Query_lora_b_states.view(batch_size,seq_len,-1,self.query_head_dim).transpose(1,2)  # [batch_zise,seq_len,num_head,query_head_dim]
        # 拆分RoPE和非RoPE部分;[batch_zise,seq_len,num_head,query_head_dim] ---->[batch_zise,seq_len,num_head,qk_nrope_head_dim]、[batch_zise,seq_len,num_head,qk_rope_head_dim]
        # Query_pass保留​​位置无关的内容特征​​，聚焦语义相似性匹配;q_rot通过RoPE注入​​位置特征​​，捕捉token的序列顺序;解耦序列顺序特征和内容语义特征
        Query_pass,Query_RoPE = torch.split(Query_states,[self.qk_nrope_head_dim,self.qk_rope_head_dim],dim=-1)
        # query将高维特征压缩为低秩特征，同时lora和rope部分采用同一个线性层进行降维，线性变换作线性组合，这个设计很精巧 ，和lora的思路如出一辙[batch_zise,seq_len,hidden_dim]---->[batch_zise,seq_len,q_lora_rank]
        # lora采用低秩分解解耦新知识特征和预训练特征；此处的设计是解耦序列顺序特征和内容特征 ;[batch_zise,seq_len,hidden_dim]----> [batch_zise,seq_len,kv_lora_rank+qk_rope_head_dim]
        Key_Value_lora_a = self.key_value_lora_a_proj(hidden_states)
        # 将位置信息和内容信息拆分；
        ## 为什么Query先升维后Split，Key先Split后升维？
        ## query是当前token的特征，需要的语义表达能力，所以在升维之后在split
        ## Key是捕捉上下文索引信息，需要上下文索引能力，所以在split之后升维
        ##  [batch_zise,seq_len,kv_lora_rank+qk_rope_head_dim]--->[batch_zise,seq_len,kv_lora_rank]、[batch_zise,seq_len,qk_rope_head_dim]
        Key_pass,Key_RoPE = torch.split(Key_Value_lora_a,[self.kv_lora_rank,self.qk_rope_head_dim],dim=-1)
        # 归一化；[batch_zise,seq_len,kv_lora_rank]---->[batch_zise,seq_len,kv_lora_rank]
        Key_pass = self.key_value_lora_layernorm(Key_pass)
        # 仅对内容信息进行升维;[batch_zise,seq_len,kv_lora_rank]----> [batch_zise,seq_len,num_kv_heads *(qk_nrope_head_dim + value_head_dim)] 
        Key_pass = self.key_value_lora_b_proj(Key_pass)
        # 矩阵变换[batch_zise,seq_len,num_kv_heads *(qk_nrope_head_dim + value_head_dim)] ---> [batch_zise,seq_len,num_kv_heads ,qk_nrope_head_dim + value_head_dim] --> [batch_zise,num_kv_heads ,seq_len,qk_nrope_head_dim + value_head_dim] 
        Key_pass = Key_pass.view(batch_size,seq_len,-1,self.qk_nrope_head_dim+self.value_head_dim).transpose(1,2)
        # [batch_zise,num_kv_heads ,seq_len ,qk_nrope_head_dim + value_head_dim] ----> [batch_zise,num_kv_heads ,seq_len,qk_nrope_head_dim] 、[batch_zise,num_kv_heads ,seq_len ,value_head_dim] 
        Key_pass,Value_states = torch.split(Key_pass,[self.qk_nrope_head_dim,self.value_head_dim],dim=-1)
        # [batch_zise,seq_len,qk_rope_head_dim]---> [batch_zise,1,seq_len,qk_rope_head_dim]
        Key_RoPE = Key_RoPE.view(batch_size,1,seq_len,self.qk_rope_head_dim)

        sin,cos = self.rotaryemb(Key_RoPE,seq_len)

 
        Query_RoPE = get_rotary_pos_emb(Query_RoPE,sin,cos)
        Key_RoPE = get_rotary_pos_emb(Key_RoPE,sin,cos)
        # [batch_zise,1,seq_len,qk_rope_head_dim] ---> [batch_zise,num_kv_heads,seq_len,qk_rope_head_dim]
        ## Key_pass:[batch_zise,num_kv_heads,seq_len ,qk_nrope_head_dim] 和Key_pass前三维一致,
        Key_RoPE = Key_RoPE.expand(batch_size,self.num_kv_heads,seq_len,-1) #Key_pass： [batch_zise,num_kv_heads，seq_len]
        # [batch_zise,seq_len,num_head,qk_nrope_head_dim]+[batch_zise,seq_len,num_head,kv_lora_rank+qk_rope_head_dim]---->[batch_zise,seq_len,num_head,qk_nrope_head_dim+qk_rope_head_dim]
        Query_states = torch.cat([Query_pass,Query_RoPE],dim=-1)
        # [batch_zise,num_kv_heads ,seq_len,qk_nrope_head_dim] + [batch_zise,num_kv_heads,seq_len,qk_rope_head_dim]----> [batch_zise,num_kv_heads,seq_len,qk_nrope_head_dim+qk_rope_head_dim]
        Key_states = torch.cat([Key_pass,Key_RoPE],dim=-1)
        # [batch_zise,num_kv_heads ,seq_len,qk_nrope_head_dim+qk_rope_head_dim] -->[batch_zise,num_kv_heads,1,seq_len,qk_nrope_head_dim+qk_rope_head_dim]-->[batch_zise,num_kv_heads,num_kv_groups,seq_len,qk_nrope_head_dim+qk_rope_head_dim]-->[batch_zise,num_kv_heads*num_kv_groups,seq_len,qk_nrope_head_dim+qk_rope_head_dim]
        Key_states = Key_states.unsqueeze(2).expand(-1,-1,self.num_kv_groups,-1,-1).reshape(batch_size,self.num_kv_heads*self.num_kv_groups,seq_len,-1)
        Value_states = Value_states.unsqueeze(2).expand(-1,-1,self.num_kv_groups,-1,-1).reshape(batch_size,self.num_kv_heads*self.num_kv_groups,seq_len,-1)

        # 注意力分数
        ## [batch_zise,num_head,seq_len,qk_nrope_head_dim+qk_rope_head_dim]*[batch_zise,num_kv_heads*num_kv_groups,qk_nrope_head_dim+qk_rope_head_dim,seq_len]-->[batch_zise,num_head,seq_len,seq_len]
        attention_score = torch.matmul(Query_states,Key_states.transpose(-1,-2))
        scale = 1/math.sqrt(self.query_head_dim) 
        if self.mscale_dim:
            if scale<=1:
                mscale = 1
            else:
                mscale = 0.1*self.mscale_dim * math.log(self.scaling_factor) + 1.0
            scale = scale*mscale*mscale        
        attention_score = attention_score*scale  # query_head_dim = qk_nrope_head_dim+qk_rope_head_dim
        if attention_mask:
            attention_score = attention_score.masked_fill(attention_mask==0,float("-inf"))
        attention_weight = F.softmax(attention_score,dim=-1)
        if self.dropout_rate:
            attention_weight = self.atten_dropout(attention_weight)
        # [batch_zise,num_head,seq_len,seq_len]*[batch_zise,num_kv_heads*num_kv_groups,seq_len,value_head_dim]--->[batch_zise,num_head,seq_len,value_head_dim]
        attention_output = torch.matmul(attention_weight,Value_states)
        # [batch_zise,num_head,seq_len,value_head_dim]--->[batch_zise,seq_len,num_head,value_head_dim]---> [batch_zise,seq_len,num_head*value_head_dim]
        attention_output = attention_output.transpose(1,2).contiguous().reshape(batch_size,seq_len,-1)
        attention_output = self.output_proj(attention_output)
        return attention_output,attention_weight
    

        


        






        
    












