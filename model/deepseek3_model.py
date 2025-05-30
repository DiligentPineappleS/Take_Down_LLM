import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
import math
import transformers
from transformers.modeling_utils import  PreTrainedModel
from transformers.generation import GenerationMixin

class RotaryEmbedding(nn.Module):
    def __init__(self,hidden_dim, max_len = 2048,base = 100000):
        super().__init__()    
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        inv_freqs = 1/(base**(torch.arange(0,self.hidden_dim,2).float()/self.hidden_dim))
        self.register_buffer("inv_freqs",inv_freqs,persistent=False)

        self._repo_position_emb(self.max_len)
    def _repo_position_emb(self,max_len):
        seq_cache = torch.arange(max_len,dtype=torch.get_default_dtype())
        seq_emb = torch.outer(self.inv_freqs,seq_cache)
        sin_cache = seq_emb.sin()
        cos_cache = seq_emb.cos()
        self.register_buffer("cos_cache",sin_cache,persistent=False)
        self.register_buffer("sin_cache",cos_cache,persistent=False)
    def forward(self,x,seq_len):
        if seq_len>self.max_len:
            self._repo_position_emb(seq_len)
        return (self.sin_cache[:seq_len].to(dtype=x.dtype),self.cos_cache[:seq_len].to(dtype=x.dtype))
    
def retate_half(x):
    hidden_dim = x.size(-1)
    x1 = x[...,:hidden_dim/2]
    x2 = x[...,hidden_dim/2:]
    return torch.cat([-x2,x1])

def get_rotary_pos_emb(x,sin,cos,unsqueeze_dim =1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x*cos) + retate_half(x)*sin


def get_rotary_pos_interleave_emb(x,sin,cos,unsqueeze_dim =1):
    batch_size,num_heads,seq_len,hidden_dim = x.shape
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x = x.reshape(batch_size,num_heads,seq_len,hidden_dim//2).transpose(4,3).reshape(batch_size,num_heads,seq_len,hidden_dim)
    return (x*cos) + retate_half(x)*sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim,eps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = torch.ones(self.hidden_dim)
    def forward(self,x:torch.Tensor):
        x_ = x*torch.rsqrt(x.pow(2).mean(-1,keepdim = True)+self.eps)
        return self.gamma*x_.type_as(x)
    
class Deepseek3Attention(nn.Module):
    def __init__(self,hidden_dim, query_head_dim,value_head_dim,num_kv_heads,num_heads,max_len,q_lora_rank,kv_lora_rank,
                 qk_rope_head_dim,qk_nrope_head_dim,keyvalue_head_dim,attention_bias,scaling_factor,is_rope_interleave,mscale_dim,dropout_rate=None,eps=1e-6,base=10000):
        super(Deepseek3Attention,self).__init__()
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
        self.rotaryemb = RotaryEmbedding(hidden_dim=self.hidden_dim,max_len=self.max_len,base=self.base)
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

        if self.is_rope_interleave:
            Query_RoPE = get_rotary_pos_interleave_emb(Query_RoPE,sin,cos)
            Key_RoPE = get_rotary_pos_interleave_emb(Key_RoPE,sin,cos)
        else:
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
    


class MLP(nn.Module):
    def __init__(self, hidden_dim,intermediate_size,activation_func):
        super(MLP,self).__init__()
        self.hidden_dim = hidden_dim
        # 可以设置为动态的；
        self.intermediate_size = intermediate_size 
        self.activation_func = activation_func
        self.gatenet = nn.Linear(self.hidden_dim,self.intermediate_size,bias=False)
        self.W0 = nn.Linear(self.hidden_dim,self.intermediate_size,bias=False)
        self.W1 = nn.Linear(self.intermediate_size,self.hidden_dim,bias=False)
        self.activation_func = get_activation(activation_func)
    def forward(self,x):
        gatenet_output = self.gatenet(x)
        gatenet_weight = self.activation_func(gatenet_output)
        W0_output = self.W0(x)
        W1_output = self.W1(gatenet_weight*W0_output)
        return W1_output
  

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, hidden_dim,num_expert_per_token,num_router_expert,num_group,group_topk,routed_scaling_factor,norm_topk_prop):
        super(MLP,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_router_expert = num_router_expert # 路由专家的个数
        self.num_expert_per_token = num_expert_per_token # 每个toekn选择的专家的个数
        self.num_group = num_group # 专家分组数
        self.group_topk = group_topk # 每个组中选择的专家个数
        self.norm_topk_prop = norm_topk_prop # 是否进行归一化
        self.routed_scaling_factor = routed_scaling_factor
        # 用Linear层替代原始参数
        self.router_weight = nn.Linear(self.hidden_dim,self.num_router_expert,bias=True )   
         # 冻结原始缓冲区的定义
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    def init_parameters(self):
        """参数初始化适配"""
        # 原始weight的初始化逻辑（如Xavier）
        nn.init.xavier_normal_(self.router_weight.weight.T)  # 注意转置
        # 将e_score_correction_bias迁移到Linear的bias
        with torch.no_grad():
            self.router_weight.bias.copy_(self.e_score_correction_bias)
    
    @torch.no_grad()
    def get_topk_index(self,scores):
        # 将分数进行合并，按照专家个数；[batch_size,seq_len,num_router_expert]
        scores_for_choice = scores.view(-1,self.num_router_expert)+self.e_score_corrections_bias.unsqueeze(0)
        group_score = (scores_for_choice.view(-1,self.num_group,self.num_router_expert/self.num_group).topk(2,dim=-1)[0].sum(dim=-1))
        group_idx = torch.topk(group_score,k=self.group_topk,dim =-1,sorted=False)[1]
        group_mask = torch.zeros_like(group_score)
        group_score.scatter_(1,group_idx,1)
        score_mask = (group_mask.unsqueeze(-1).expand(-1,self.num_group,self.num_router_expert//self.num_group).reshape(-1,self.num_router_expert))
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(),0.0)
        topk_index = torch.topk(scores_for_choice,k = self.topk,dim=-1,sorted=False)[1]
        return topk_index
    def forward(self,hidden_states):
        ### 整体流程：1. 选择路由；2. 计算路由分数；3.路由分数分组；4.每组提取topk2的分数求和；5.提取分数前group_topk的组；6.
        # 合并；[batch_size,seq_len,hidden_dim]--->[batch_size*seq_len,hidden_dim]
        hidden_states = hidden_states.view(-1,self.hidden_dim)
        # 计算路由[batch_size*seq_len,hidden_dim]--->[batch_size*seq_len,num_router_expert]
        router_logits = self.router_weight(hidden_states.type(torch.float32))
        # 计算分数 [batch_size*seq_len,num_router_expert]
        router_score = F.sigmoid(router_logits)
        # topk专家提取 [batch_size*seq_len,num_router_expert]-->[batch_size*seq_len,num_group,num_router_expert]
        ## 专家分组,将专家分组成[num_group, experts_per_group]，取每组前2名得分求和
        group_score = router_score.view(-1,self.num_group,self.num_router_expert/self.num_group)
        ## 提取分组中分数topk,并求和
        ## 组内得分求和：[batch_size*seq_len, num_group]
        group_score = group_score.topk(2,dim=-1)[0].sum(dim=-1) 
        ## 从所有num_group组中仅保留最重要的topk_group个组，然后获取分组topk的索引,选择得分最高的topk_group个组; [batch*seq_len, topk_group]--->[topk_group]每个元素是一个整数，表示被选中的专家组在num_group专家分组中的索引
        group_score_topk_index = torch.topk(group_score,k = self.group_topk,dim=-1,sorted=False)[1]
        ## 创建与group_score_topk形状相同的全零张量  [batch*seq_len, num_group];
        group_mask = torch.zeros_like(group_score)
        ## 构建mask矩阵,专家索引映射位置设置为1，[batch*seq_len, num_group]
        group_score.scatter_(dim=1,index=group_score_topk_index,src=1)
        ## 将mask还原为和router_score相同的矩阵，[batch*seq_len, num_group]--->[batch*seq_len, num_group,1]--->[batch*seq_len, num_group,num_router_expert/num_group]--->[batch*seq_len, num_router_expert]
        score_mask = group_mask.unsqueeze(-1).expand(-1,self.num_group,self.num_router_expert/self.num_group).reshape(-1,self.num_router_expert)

        router_score_mask = router_score.masked_fill(score_mask==0,torch.float(0))
        topk_index = torch.topk(router_score_mask,k = self.num_expert_per_token,dim=-1,sorted=False)[1]
        topk_weights = router_score.gather(1, topk_index)
        if self.norm_topk_prop:
            topk_weights = topk_weights/(topk_weights.sum(dim=-1,keepdim=True)+1e-20)
        topk_weights = topk_weights*self.routed_scaling_factor
        return topk_index,topk_weights


class MOE(nn.Module):
    def __init__(self, hidden_dim,num_expert,expert_intermediate_size,topk,shared_expert_intermediate_size,activation_func,norm_topk,
                 num_share_experts,num_group,group_topk,norm_topk_prop,routed_scaling_factor,num_expert_per_token):
        super(MOE,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert 
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.activation_func = activation_func
        self.topk = topk
        self.norm_topk = norm_topk
        self.num_share_experts = num_share_experts
        self.num_expert_per_token = num_expert_per_token # 每个toekn选择的专家的个数
        self.num_group = num_group # 专家分组数
        self.group_topk = group_topk # 每个组中选择的专家个数
        self.norm_topk_prop = norm_topk_prop # 是否进行归一化
        self.routed_scaling_factor = routed_scaling_factor
        # 
        self.gatenet = DeepseekV3TopkRouter(self.hidden_dim,self.num_expert_per_token,self.num_expert, self.num_group, self.group_topk, self.routed_scaling_factor, self.nnorm_topk_prop)
        self.expertnets = nn.ModuleList([MLP(self.hidden_dim,self.expert_intermediate_size,self.activation_func) for i in range(self.num_expert)])
        
        self.share_expertnet = MLP(self.hidden_dim,self.shared_expert_intermediate_size*self.num_share_experts,self.activation_func)
    def forward(self,hidden_states):
        res = hidden_states
        batch_size,seq_len,hidden_dim = hidden_states.shape
        gate_index,gate_weight = self.gatenet(hidden_states)
        hidden_states = hidden_states.view(-1,hidden_dim)
        final_hidden_states = torch.zeros_like(hidden_states,dtype=gate_weight.dtype)
        expert_mask = torch.nn.functional.one_hot(gate_index,num_classes=self.num_expert).permute(2,0,1)
        for expertid in range(len(self.expertnets)):
            expertnet = self.expertnets[expertid]
            mask = expert_mask[expertid]
            token_indexs,weight_index = torch.where(mask)
            if token_indexs.numel()>0:
                expert_weight = gate_weight[token_indexs,weight_index]
                expert_input = hidden_states][token_indexs]
                expert_output = expertnet(expert_input)
                weight_output = expert_output*expert_weight.unsqueeze(-1)
                final_hidden_states.index_add_(0,token_indexs,weight_output)
        hidden_states = final_hidden_states.type(hidden_states.dtype)
        hidden_states = hidden_states + self.share_expertnet(res)
        return hidden_states



class  Deepseek3Decoderlayer(PreTrainedModel):
    def __init__(self,layer_id,hidden_dim,max_len,num_heads,num_kv_heads,num_expert,expert_intermediate_size,
                 topk,shared_expert_intermediate_size,activation_func,norm_topk,dropout_rate,eps,base, k_dense,
                 num_share_experts,num_group,group_topk,norm_topk_prop,routed_scaling_factor,num_expert_per_token):
        super(Deepseek3Decoderlayer,self).__init__()
        """
        模型Decoder架构：
            (hidden_states)---> [RMSNorm]---> [Attention]---> (hidden_states+res)---> [RMSNorm]---> [MLP]
        """
        self.layer_id = layer_id
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert 
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.activation_func = activation_func
        self.topk = topk
        self.norm_topk = norm_topk
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps   
        self.max_len = max_len     
        self.base = base
        self.dropout_rate = dropout_rate
        self.k_dense = k_dense
        # 1. RMSNorm
        self.atten_rmsnorm = RMSNorm(self.hidden_dim,self.eps)
        # 2. Attention
        self.Attention = Deepseek3Attention(self.hidden_dim,self.num_kv_heads,self.num_heads,self.max_len,self.dropout_rate,self.base)
        # 3. RMSNorm
        self.mlp_rmsnorm = RMSNorm(self.hidden_dim,self.eps)
        # 4. MLP 这里就是前K层使用MLP，后K层使用MOE；这样做的目的是前K层的用来提取特征，主要做特征表达；后K层用MOE来提升模型的表达能力；
        if self.layer_id>=self.k_dense:
            self.mlp = MOE(hidden_dim,num_expert,expert_intermediate_size,topk,shared_expert_intermediate_size,activation_func,norm_topk,
                 num_share_experts,num_group,group_topk,norm_topk_prop,routed_scaling_factor,num_expert_per_token)
        else:
            self.mlp = MLP(self.hidden_dim,self.expert_intermediate_size,self.activation_func)
 
    def forward(self,hidden_states,attention_mask,output_attention):
        res = hidden_states
        hidden_states = self.atten_rmsnorm(hidden_states)
        hidden_states,attention_weight = self.Attention(hidden_states,attention_mask)
        hidden_states = res + hidden_states
        res = hidden_states
        hidden_states = self.mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = res + hidden_states
        output = (hidden_states,)
        if output_attention:
            outputs += (attention_weight,)
        return output



class DeepseekV3Model(PreTrainedModel,GenerationMixin):
    def __init__(self,volab_size,num_layers,hidden_dim,max_len,num_heads,num_kv_heads,num_expert,expert_intermediate_size,
                topk,shared_expert_intermediate_size,activation_func,norm_topk,dropout_rate,eps,base, k_dense,
                num_share_experts,num_group,group_topk,norm_topk_prop,routed_scaling_factor,num_expert_per_token):
        super(DeepseekV3Model,self).__init__()
        self.num_layers = num_layers
        self.volab_size = volab_size
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert 
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.activation_func = activation_func
        self.topk = topk
        self.norm_topk = norm_topk
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps   
        self.max_len = max_len     
        self.base = base
        self.dropout_rate = dropout_rate
        self.k_dense = k_dense
        self.emb = nn.Embedding(self.volab_size,self.hidden_dim)
        self.layers = nn.ModuleList([Deepseek3Decoderlayer(layer_id,hidden_dim,max_len,num_heads,num_kv_heads,num_expert,expert_intermediate_size,
                 topk,shared_expert_intermediate_size,activation_func,norm_topk,dropout_rate,eps,base, k_dense,
                 num_share_experts,num_group,group_topk,norm_topk_prop,routed_scaling_factor,num_expert_per_token) for layer_id in range(num_layers)])
        self.DecoderRSMnorm = RMSNorm(self.hidden_dim,self.eps)
        self.LMRSMnorm = RMSNorm(self.hidden_dim,self.eps)

        self.LM_Weight = nn.Linear(self.hidden_dim, self.volab_size, bias=False)

        self.apply(self._init_weights)
        cached_mask = torch.full((1, 1, self.max_len, self.max_len), float("-inf"))
        cached_mask = torch.triu(cached_mask, diagonal=1)
        # 注册为模型的缓冲区
        self.register_buffer("cached_mask", cached_mask)

    def _init_weights(self, module):
        std = 0.01
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
            
    def forward(self,input_ids,target,output_attention):
        hidden_states = self.emb(input_ids)
        batch_size,seqlen = input_ids.shape
        # 动态生成因果注意力掩码 (优化内存效率)
        if not hasattr(self, 'cached_mask') or self.cached_mask.size(3) < seqlen:
            # 创建下三角因果掩码 (1表示允许注意力，0表示屏蔽)
            self.cached_mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, device=input_ids.device))
        atten_all_output = ()
        attention_mask = self.cached_mask[:, :, :seqlen, :seqlen]
        for layer in self.layers:
            decoderlayer_ouput = layer(hidden_states,attention_mask,output_attention)
            hidden_states = decoderlayer_ouput[0]
            if output_attention:
                atten_all_output += (decoderlayer_ouput[1],)
        hidden_states = self.LMRSMnorm(hidden_states)
        if target:# 下一个词
            logits = self.LM_Weight(hidden_states)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                             target.view(-1), 
                                             ignore_index=-1)
        else:
            logits = self.LM_Weight(hidden_states[:, [-1], :]) 
            self.last_loss = None
        return logits        

        

        

            
            