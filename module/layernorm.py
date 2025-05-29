
import torch
import torch.nn as nn


# 归一化没有什么好说的，按照公式写代码即可，不过batchnorm的代码逻辑要自行梳理一下因为不是重点，这个

class BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(num_features))  # 平移参数
        self.register_buffer("running_mean", torch.zeros(num_features))  # 移动平均均值
        self.register_buffer("running_var", torch.ones(num_features))     # 移动平均方差

    def forward(self, x: torch.Tensor):
        # 输入x形状: (batch_size, num_features, ...)（如CNN中的(N,C,H,W)）
        if self.training:
            # 训练模式：计算当前批次的均值/方差，并更新移动平均
            dims = (0,) + tuple(range(2, x.dim()))  # 沿批量+空间维度统计（如CNN的H,W）
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            with torch.no_grad():  # 更新全局统计量
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用预存的移动平均
            mean = self.running_mean
            var = self.running_var

        # 归一化与仿射变换
        x_normalized = (x - mean.view(1, -1, *([1]*(x.dim()-2)))) / torch.sqrt(var.view(1, -1, *([1]*(x.dim()-2))) + self.eps)
        return self.gamma.view(1, -1, *([1]*(x.dim()-2))) * x_normalized + self.beta.view(1, -1, *([1]*(x.dim()-2)))


# LayerNorm
## LayerNorm 对每个样本的所有特征维度进行归一化，
class LayerNorm(nn.Module):
    def __init__(self, dim,eps):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    def forward(self,x:torch.Tensor):
        mean = x.mean(dim=-1,keepdim = True) # 沿特征维度（dim=-1）统计
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        temp_x =(x - mean) /torch.sqrt(var + self.eps) 
        return self.gamma * temp_x +self.beta




## RMSNorm 的核心思想是 ​​仅对输入向量进行缩放（Scale）而不进行平移（Shift）
## 优点：1、简化计算、降低计算资源消耗和内存占用；这点从公式可发现；
     ## 2、 采用均方根代替方差，计算更稳定；（方差是每个值与均值的偏离平方的均值，它对于异常值更敏感；均方根原始值的平方均值再开平方，不依赖均值；在大语言模型中当语义相差较大时，rms采用均方根进行归一化相对会让整个分布的波动更小）

class RMSNorm(nn.Module):
    def __init__(self,dim,eps):
        super(RMSNorm).__init__()
        self.eps = eps
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        return self._norm(x.float()).type_as(x)*self.gamma







