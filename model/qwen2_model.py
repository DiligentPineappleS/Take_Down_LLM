import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
import math
import transformers


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



class RMSNorm(nn.Module):
    def __init__(self, hidden_dim,eps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = torch.ones(self.hidden_dim)
    def forward(self,x:torch.Tensor):
        x_ = x*torch.rsqrt(x.pow(2).mean(-1,keepdim = True)+self.eps)
        return self.gamma*x_.type_as(x)
    
class QWen2Attention(nn.Module):
    def __init__(self, d_model,num_kv_heads,num_heads,max_len,dropout_rate=None,base=10000):
        super(QWen2Attention,self).__init__()
        assert d_model%num_heads==0,"d_model must be num_heads*number,number is int"
        assert num_heads%num_kv_heads == 0 ,"num_heads must be num_groups*num_kv_heads,num_kv_heads is int"
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.base = base
        self.hidden_dim = self.d_model//self.num_heads
        self.num_groups = self.num_heads//self.num_kv_heads
        
        self.WK = nn.Linear(self.d_model,self.hidden_dim * self.num_kv_heads,bias=True)
        self.WV = nn.Linear(self.d_model,self.hidden_dim * self.num_kv_heads,bias=True)
        self.WQ = nn.Linear(self.d_model,self.d_model,bias=True)
        self.WO = nn.Linear(self.d_model,self.d_model,bias=True)
        if dropout_rate:
            self.atten_dropout = nn.Dropout(self.dropout_rate)
        self.rotaryemb = RotaryEmbedding(hidden_dim=self.hidden_dim,max_len=self.max_len,base=self.base)


    def forward(self,query:torch.Tensor,attention_mask):
        batch_size,seq_len,hidden_dim = query.shape

        Q_states = self.WQ(query)   # [batch_zise,seq_len,d_model]
        K_states = self.WK(query)   # [batch_zise,seq_len,hidden_dim * num_kv_heads]
        V_states = self.WV(query)   # [batch_zise,seq_len,hidden_dim * num_kv_heads]

        # query变换;[batch_zise,seq_len,d_model]-->[batch_zise,seq_len,num_heads,hidden_dim]-->[batch_zise,num_heads,seq_len,hidden_dim]
        Query = Q_states.view(batch_size,seq_len,self.num_heads,self.hidden_dim).transpose(1,2)

        # key\value多头;[batch_zise,seq_len,d_model]-->[batch_zise,seq_len,num_kv_heads,hidden_dim]-->[batch_zise,num_kv_heads,seq_len,hidden_dim]
        Key = K_states.view(batch_size,seq_len,self.num_kv_heads,self.hidden_dim).transpose(1,2)
        Value = V_states.view(batch_size,seq_len,self.num_kv_heads,self.hidden_dim).transpose(1,2)

        # 计算位置编码；
        sin,cos = self.rotaryemb(Key,Key.size(2))
        Key = get_rotary_pos_emb(Key,sin,cos)
        Value = get_rotary_pos_emb(Value,sin,cos)

        # key\value共享；[batch_zise,num_kv_heads,seq_len,hidden_dim]--->[batch_zise,num_kv_heads,1,seq_len,hidden_dim]--->[batch_zise,num_kv_heads,num_groups,seq_len,hidden_dim]
        Key = Key.unsqueeze(2).expand(-1,-1,self.num_groups,-1,-1)
        Value = Value.unsqueeze(2).expand(-1,-1,self.num_groups,-1,-1)
        # key/value处理为多头注意力结构；[batch_zise,num_kv_heads,num_groups,seq_len,hidden_dim]--->[batch_zise,num_kv_heads*num_groups,seq_len,hidden_dim]==query:[batch_zise,num_heads,seq_len,hidden_dim]
        Key = Key.reshape(batch_size,self.num_groups*self.num_kv_heads,seq_len,self.hidden_dim)
        Value = Value.reshape(batch_size,self.num_groups*self.num_kv_heads,seq_len,self.hidden_dim)
        # 计算注意力分数；[batch_zise,num_heads,seq_len,hidden_dim] *[batch_zise,num_heads,hidden_dim，seq_len]---> [batch_zise,num_heads,seq_len,seq_len]
        attention_score = torch.matmul(Query,Key.transpose(-1,-2))/math.sqrt(self.hidden_dim)
        if attention_mask:
            # 掩码处理
            attention_score = attention_score.masked_fill(attention_mask==0,float("-inf"))
        # 计算注意力权重；[batch_zise,num_heads,seq_len,seq_len]-->[batch_zise,num_heads,seq_len,seq_len]
        attention_weight = F.softmax(attention_score,dim=-1)
        if self.dropout_rate:
            # dropout ;
            attention_weight = self.atten_dropout(attention_weight)
        # 计算注意力输出；[batch_zise,num_heads,seq_len,seq_len] *[batch_zise,num_heads,seq_len,hidden_dim,]---> [batch_zise,num_heads,seq_len,hidden_dim]
        attention_output = torch.matmul(attention_weight,Value)
        # 还原输出；[batch_zise,num_heads,seq_len,hidden_dim]--->[batch_zise,seq_len,num_heads,hidden_dim]---->[batch_zise,seq_len,num_heads*hidden_dim]
        attention_output = attention_output.transpose(1,2).contiguous().reshape(batch_size,seq_len,self.d_model)
        # 整合信息；
        output = self.WO(attention_output)
        return output,attention_weight


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
  

class  QwenDecoderlayer(nn.Module):
    def __init__(self,layer_id,hidden_dim,max_len,num_heads,num_kv_heads,num_expert,expert_intermediate_size,
                 topk,shared_expert_intermediate_size,activation_func,norm_topk,dropout_rate,eps,base,mode):
        super(QwenDecoderlayer,self).__init__()
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
        # 1. RMSNorm
        self.atten_rmsnorm = RMSNorm(self.hidden_dim,self.eps)
        # 2. Attention
        self.Attention = QWen2Attention(self.hidden_dim,self.num_kv_heads,self.num_heads,self.max_len,self.dropout_rate,self.base)
        # 3. RMSNorm
        self.mlp_rmsnorm = RMSNorm(self.hidden_dim,self.eps)
        # 4. MLP
        self.mlp = MLP(self.hidden_dim,self.expert_intermediate_size,self.activation_func)
 
    def forward(self,hidden_states,attention_mask,output_attention):
        res = hidden_states
        hidden_states = self.atten_rmsnorm(hidden_states)
        hidden_states,attention_weight = self.Attention(hidden_states,attention_mask)
        hidden_states = res + hidden_states
        hidden_states = self.mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = res + hidden_states
        output = (hidden_states,)
        if output_attention:
            outputs += (attention_weight,)
        return output


   
class QWENModel_MLP(nn.Module):
    def __init__(self,hidden_dim,max_len,num_heads,num_kv_heads,num_expert,expert_intermediate_size,
                 topk,shared_expert_intermediate_size,activation_func,norm_topk,dropout_rate,eps,base,mode,vocab_size,num_hidden_layers):
        super(QWENModel_MLP,self).__init__()
   
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
        self.mode = mode
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.base = base
        self.emb = nn.Embedding(self.vocab_size,self.hidden_dim)


        self.decoder_layer_list = nn.ModuleList([QwenDecoderlayer(i,self.hidden_dim,self.max_len,self.num_heads,self.num_kv_heads,self.num_expert,
                                                           self.expert_intermediate_size,self.topk,self.shared_expert_intermediate_size,
                                                           self.activation_func,self.norm_topk,self.dropout_rate,self.eps,self.base,self.mode) for i in range(num_hidden_layers)])
        
        self.RMSNorm = RMSNorm(self.hidden_dim,self.eps)

        self.output = nn.Linear(self.hidden_dim,self.vocab_size)
        # 将词嵌入层的权重与输出层的权重共享
        self.emb.weight = self.output.weight 
        cached_mask = torch.full((1, 1, self.max_len, self.max_len), float("-inf"))
        cached_mask = torch.triu(cached_mask, diagonal=1)
        # 注册为模型的缓冲区
        self.register_buffer("cached_mask", cached_mask)
        self.output_attention = False
        self.apply(self._init_weights)

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

    def get_mask(self, seqlen: int):
        if seqlen > self.max_len:
            return torch.tril(torch.ones(1, 1, seqlen, seqlen, device=self.device))
        return self.cached_mask[:, :, :seqlen, :seqlen]
    
    def forward(self,input_ids,targets):
        hidden_states = self.emb(input_ids)
        batch_size,seqlen = input_ids.shape
        # 动态生成因果注意力掩码 (优化内存效率)
        if not hasattr(self, 'cached_mask') or self.cached_mask.size(3) < seqlen:
            # 创建下三角因果掩码 (1表示允许注意力，0表示屏蔽)
            self.cached_mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, device=input_ids.device))
        attention_mask = self.cached_mask[:, :, :seqlen, :seqlen]

        for decoder_layer in self.decoder_layer_list:
            hidden_states = decoder_layer(hidden_states,attention_mask,self.output_attention)
        hidden_states = self.RMSNorm(hidden_states)
        if targets:# 下一个词
            logits = self.output(hidden_states)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                             targets.view(-1), 
                                             ignore_index=-1)
        else:
            logits = self.output(hidden_states[:, [-1], :]) 
            self.last_loss = None
        return logits
    


    def optimizers(self, weight_decay, learning_rate, betas):
        all_parameters = self.named_parameters()
        requires_grad_params = []
        no_requires_grad_params = []
        for i,param in all_parameters:
            if param.requires_grad:
                if param.dim()>=2:
                    requires_grad_params.append(param)
                else:
                    no_requires_grad_params.append(param)
            else:
                continue
        # 验证参数分组完整性（防止配置错误）
        total_params = len(requires_grad_params) + len(no_requires_grad_params)
        assert total_params == len(list(self.parameters())), "参数分组存在遗漏！"
    
        optim_groups = [
            {'params': requires_grad_params, 'weight_decay': weight_decay},
            {'params': no_requires_grad_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,fused=True,capturable=True)  # 支持CUDA Graph捕获 

        return optimizer


    
    












        

        
        

































        








