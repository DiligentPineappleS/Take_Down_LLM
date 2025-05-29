import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation

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
  

    
class MOE(nn.Module):
    def __init__(self, hidden_dim,num_expert,expert_intermediate_size,topk,shared_expert_intermediate_size,activation_func,norm_topk,):
        super(MOE,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert 
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.activation_func = activation_func
        self.topk = topk
        self.norm_topk = norm_topk
        # 
        self.gatenet = nn.Linear(self.hidden_dim,self.num_expert,bias=False)
        self.expertnets = nn.ModuleList([MLP(self.hidden_dim,self.expert_intermediate_size,self.activation_func) for i in range(self.num_mlp_layers)])
        
        self.share_gatenet = nn.Linear(self.hidden_dim,1,bias=False)
        self.share_expertnet = MLP(self.hidden_dim,self.shared_expert_intermediate_size,self.activation_func)
    def forward(self,hidden_states):
        # hidden_states变换；[batch_size,seq_len,hidden_dim] ----> [batch_size*seq_len,hidden_dim] 
        ## 也就是说这里把所有的token合并到一起了
        batch_size,seq_len,hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # 计算每个token的选择专家的权重;[batch_size*seq_len,hidden_dim]-->[batch_size * seq_len, num_experts]​-->[batch_size * seq_len, num_experts]​​ 
        gate_output = self.gatenet(hidden_states)
        gate_weight = nn.Softmax(gate_output,dim=-1)
        # 根据权重选择topk个专家;[batch_size * seq_len, num_experts]​​ --->[batch_size * seq_len,self.topk]
        topk_gate_weights,topk_gate_indexs = gate_weight.topk(self.topk,dim=-1)
        # 对选择出的专家权重进行标准化：[batch_size * seq_len,self.topk]--->[batch_size * seq_len,self.topk]
        if self.norm_topk:
            topk_gate_weights = topk_gate_weights/topk_gate_weights.sum(dim=1,keepdim=True)

        # 采用独热编码的方式标记每个token选择的专家；[batch_size * seq_len,self.topk]-->[ batch_size * seq_len,top_k,num_experts]--> [num_experts, top_k, batch_size * seq_len]
        ## 将在token维度的专家索引转换为专家维度的token索引；也就是说，原本gate_weight是根据在每个token选择topk个专家，
        ## 这里gate_weight的输出就是每个token对应的专家的ID;如：[[0,1],[1,3],[0,3],....,[token(seq_len)]]===>token 0:对应的专家0和专家1、toekn 1:对应专家1和专家3...
        ## 到这里采用onehot编码，就是把他转换为在专家维度的编码，使用二进制表示；即：[[1,0,1,...,seq_len],[1,1,0,...,seq_len],[0,1,1,...,seq_len],[num_experts]];
        ## 通俗理解，就是每个专家一个长度为seq_len的列表，列表中的数据是每个toekn选择的topk专家是否包含当前专家，包含则为1，不包含则为0；
        expert_mask = F.one_hot(topk_gate_indexs,self.num_expert).permute(2,1,0)
        #计算激活的专家；expert_mask.sum(dim = (-1,-2))每个专家维度求和选择出大于0的专家索引，输出： [num_activated_experts]
        expert_hitted = (expert_mask.sum(dim = (-1,-2))>0).nonzero()[0].tolist()

        expert_hidden_states = torch.zeros((batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

        for expertid in expert_hitted:
            expertnet = self.expertnets[expertid]
            # 提取出当前专家作用的token；因为每个token选择topk个专家，所以要返回该专家是第几个被选择和对应作用的token；idx取值范围为 [0, top_k-1]，
            idx,topx = torch.where(expert_mask(expertid))
            ## hidden_states提取出topk对应的token；
            temp_hidden = hidden_states[topx]
            # 加权融合
            temp_hidden_states = expertnet(temp_hidden) * topk_gate_weights(topx,idx,None)
            expert_hidden_states = expert_hidden_states.index_add_(0,topx,temp_hidden_states.to(hidden_states.dtype))
        shared_expert_output = self.share_expertnet(hidden_states)
        shared_expert_output = F.sigmoid(self.share_gatenet(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, gate_output


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


class DeepseekV3MOE(nn.Module):
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

