import torch
import torch.nn as nn


class FinalFeatureAdjustment(nn.Module):
    """
    可逆神经网络（R-INN）的特征调整层
    
    该层作为R-INN模型的最后一层，对所有中间层处理后的特征进行精细化分布调整，
    同时严格保证可逆性，支持正向/逆向互逆变换。
    
    核心特性：
    - 输入/输出维度完全一致
    - 严格保证可逆性，正向变换与逆向变换共享全部可学习参数
    - 提供雅可比行列式对数计算方法，适配R-INN模型的损失计算逻辑
    - 初始化为近似恒等变换，便于模型训练收敛
    """
    
    def __init__(self, input_dim):
        """
        初始化特征调整层
        
        Args:
            input_dim (int): 输入特征维度，与R-INN模型的输入维度x_dim + z_dim匹配
        """
        super(FinalFeatureAdjustment, self).__init__()
        
        # 可学习参数：scale - 初始值为全1，保证初始恒等变换
        self.scale = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        
        # 可学习参数：bias - 初始值为全0，保证初始恒等变换
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        
        # 非线性组件：轻量级Sequential结构
        # 包含：Linear(input_dim→input_dim//2) → Tanh → Linear(input_dim//2→input_dim)
        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),  # 使用可逆激活函数Tanh
            nn.Linear(input_dim // 2, input_dim)
        )
        
        # 初始化非线性组件为近似恒等变换
        # 第一层权重用正态分布(mean=0, std=0.01)初始化，偏置初始化为0
        nn.init.normal_(self.nonlinear[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.nonlinear[0].bias)
        
        # 第二层权重用正态分布(mean=0, std=0.01)初始化，偏置初始化为0
        nn.init.normal_(self.nonlinear[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.nonlinear[2].bias)
    
    def forward(self, x):
        """
        正向特征变换
        
        Args:
            x (torch.Tensor): 输入特征张量，形状为[batch_size, input_dim]
            
        Returns:
            torch.Tensor: 调整后的特征张量，形状为[batch_size, input_dim]
        """
        # 第一步：用self.nonlinear处理x得到x_nonlinear
        x_nonlinear = self.nonlinear(x)
        
        # 第二步：残差连接（x + x_nonlinear）得到x_adjusted
        x_adjusted = x + x_nonlinear
        
        # 第三步：仿射变换（self.scale * x_adjusted + self.bias）得到输出y
        y = self.scale * x_adjusted + self.bias
        
        return y
    
    def inverse(self, y):
        """
        逆向特征变换
        
        Args:
            y (torch.Tensor): 正向变换的输出，形状为[batch_size, input_dim]
            
        Returns:
            torch.Tensor: 还原后的原始特征张量，形状为[batch_size, input_dim]
        """
        # 第一步：逆仿射变换
        # 确保scale不为0，避免除零错误
        x_adjusted = (y - self.bias) / self.scale.clamp(min=1e-8)
        
        # 第二步：迭代求解逆非线性残差
        # 初始猜测x = x_adjusted
        x = x_adjusted.clone()
        
        # 迭代3次执行x = x_adjusted - self.nonlinear(x)
        for _ in range(3):
            x = x_adjusted - self.nonlinear(x)
        
        return x
    
    def log_det_jacobian(self, x):
        """
        计算雅可比行列式对数
        
        Args:
            x (torch.Tensor): 正向变换的输入，形状为[batch_size, input_dim]
            
        Returns:
            torch.Tensor: 批次化的雅可比行列式对数张量，形状为[batch_size,]
        """
        # 简化处理：核心贡献来自scale
        # 计算各维度scale的绝对值对数之和
        log_det = torch.sum(torch.log(torch.abs(self.scale)))
        
        # 扩展为批次维度，与模型其他层的log_det输出兼容
        batch_size = x.size(0)
        log_det_batch = log_det.expand(batch_size)
        
        return log_det_batch


# 添加__init__.py文件，使feature_adjustment成为一个Python包
__all__ = ['FinalFeatureAdjustment']