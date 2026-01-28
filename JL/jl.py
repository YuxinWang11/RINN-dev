# jl/jl.py （替换现有 JLLayer 的 log_det_jacobian 实现）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

class JLLayer(nn.Module):
    def __init__(self, dim: int, orthogonal_init: bool = True, use_weight_norm: bool = False):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim, bias=True)
        if orthogonal_init:
            nn.init.orthogonal_(self.linear.weight)
        else:
            nn.init.xavier_uniform_(self.linear.weight)

        # 可选：是否使用 weight_norm（默认 False，除非你确实需要）
        if use_weight_norm:
            self.linear = parametrizations.weight_norm(self.linear, name='weight', dim=0)

        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def inverse(self, x_prime: torch.Tensor) -> torch.Tensor:
        W = self.linear.weight  # 实际用于前向的权重
        b = self.linear.bias
        return F.linear(x_prime - b, W.T, bias=None) #输出结果：形状(batch_size, dim)（比如(32, 200)

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算当前权重矩阵的 log|det(W)| 并扩展成 batch 向量返回。
        若矩阵奇异（det=0），会把对应 logabsdet 用一个很小的数替代以避免 -inf。
        
        若训练过程中始终保持正交性 ，这个实现和原来的代码就是等价的。

        这样应该会更安全，因为它：
        1. 适用于任意矩阵（不只是正交矩阵）
        2. 能处理训练过程中矩阵性质的变化
        3. 有数值稳定性保护（处理奇异矩阵）
        """
        W = self.linear.weight
        sign, logabsdet = torch.slogdet(W)  # 返回 (sign符号, logabsdet对数行列式)，这是一值
        # 如果发现 sign==0 表示矩阵奇异（det == 0）
        if torch.any(sign == 0):
            # 记录警告（可选），并把那部分设为很小的 logabsdet
            # 这里用 -1e6 作为惩罚值，避免 -inf 导致数值问题
            # torch.full_like，生成一个新张量，其形状、数据类型、存储设备（CPU / GPU）完全和logabsdet一致
            logabsdet = torch.where(sign == 0, torch.full_like(logabsdet, -1e6), logabsdet)
        return logabsdet.expand(x.shape[0])#扩展为一组数，为什么？？？？？？
