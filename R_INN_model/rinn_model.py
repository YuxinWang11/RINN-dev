import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actnorm.actnorm import ActNorm1d
from JL.jl import JLLayer
from realnvp.realnvp import RealNVP
from feature_adjustment.feature_adjustment import FinalFeatureAdjustment


class RINNBlock(nn.Module):
    """
    R-INN 论文中的基本构建块：按照 ActNorm → RealNVP → JL 层的顺序组合
    实现可逆变换和雅可比行列式计算
    """
    def __init__(self, input_dim, hidden_dim=10, num_stages=4, num_cycles_per_stage=2,
                 ratio_toZ_after_flowstage=0.3, ratio_x1_x2_inAffine=0.25):
        """
        初始化 RINNBlock
        参数:
            input_dim: 输入特征维度
            hidden_dim: RealNVP 中 MLP 的隐藏层维度
            num_stages: RealNVP 中的流阶段数量
            num_cycles_per_stage: 每个流阶段中的内部循环次数
            ratio_toZ_after_flowstage: 每个流阶段后进入z输出的比例
            ratio_x1_x2_inAffine: RealNVP中Affine层x1条件部分的比例
        """
        super(RINNBlock, self).__init__()
        
        # 1. ActNorm 层：归一化输入数据
        self.actnorm = ActNorm1d(num_features=input_dim)
        
        # 2. RealNVP 层：执行可逆流变换
        self.realnvp = RealNVP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            num_cycles_per_stage=num_cycles_per_stage,
            ratio_toZ_after_flowstage=ratio_toZ_after_flowstage,
            ratio_x1_x2_inAffine=ratio_x1_x2_inAffine
        )
        
        # 3. JL 层：执行线性变换，保持可逆性
        self.jl_layer = JLLayer(dim=input_dim, orthogonal_init=True, use_weight_norm=False)
        
        
    def forward(self, x):
        """
        RINNBlock 前向变换
        输入:
            x: 输入张量，形状(batch_size, input_dim)
        输出:
            z: 变换后张量，形状(batch_size, input_dim)
            log_det: 雅可比行列式的对数总和，形状(batch_size,)
        """
        # ActNorm 变换
        x = self.actnorm(x)
        
        # RealNVP 变换 - 保存这个输出用于z_loss计算
        z_from_realnvp, log_det_realnvp = self.realnvp(x)
        
        # JL 层变换
        z = self.jl_layer(z_from_realnvp)
        
        # 计算 JL 层的雅可比行列式
        log_det_jl = self.jl_layer.log_det_jacobian(z)
        
        # 计算 ActNorm 层的雅可比行列式
        log_det_actnorm = self.actnorm.log_det_jacobian(x)
        
        # 总和所有雅可比行列式的对数
        log_det_total = log_det_realnvp + log_det_jl + log_det_actnorm
        
        return z, log_det_total
    
    def forward_with_intermediate(self, x):
        """
        RINNBlock 前向变换（带中间结果）
        输入:
            x: 输入张量，形状(batch_size, input_dim)
        输出:
            z: 变换后张量，形状(batch_size, input_dim)
            log_det: 雅可比行列式的对数总和，形状(batch_size,)
            z_from_realnvp: RealNVP层的输出（用于z_loss计算）
        """
        # ActNorm 变换
        x = self.actnorm(x)
        
        # RealNVP 变换 - 保存这个输出用于z_loss计算
        z_from_realnvp, log_det_realnvp = self.realnvp(x)
        
        # JL 层变换
        z = self.jl_layer(z_from_realnvp)
        
        # 计算 JL 层的雅可比行列式
        log_det_jl = self.jl_layer.log_det_jacobian(z)
        
        # 计算 ActNorm 层的雅可比行列式
        log_det_actnorm = self.actnorm.log_det_jacobian(x)
        
        # 总和所有雅可比行列式的对数
        log_det_total = log_det_realnvp + log_det_jl + log_det_actnorm
        
        return z, log_det_total, z_from_realnvp
    
    def inverse(self, z):
        """
        RINNBlock 逆向变换
        输入:
            z: 输入张量，形状(batch_size, input_dim)
        输出:
            x: 逆变换后张量，形状(batch_size, input_dim)
            log_det: 逆过程雅可比行列式的对数总和，形状(batch_size,)
        """
        # 逆 JL 层变换
        x = self.jl_layer.inverse(z)
        
        # 逆 RealNVP 变换
        x, log_det_realnvp = self.realnvp.inverse(x)
        
        # 逆 ActNorm 变换
        x = self.actnorm.inverse(x)
        
        # 计算逆过程的雅可比行列式（注意符号变化）
        log_det_actnorm = -self.actnorm.log_det_jacobian(x)
        log_det_jl = -self.jl_layer.log_det_jacobian(z)
        
        # 总和所有雅可比行列式的对数
        log_det_total = log_det_realnvp + log_det_jl + log_det_actnorm
        
        return x, log_det_total


class Shuffle(nn.Module):
    """
    Shuffle层：打乱输入特征的顺序，增加模型的表达能力
    """
    def __init__(self, input_dim):
        """
        初始化Shuffle层
        参数:
            input_dim: 输入特征维度
        """
        super(Shuffle, self).__init__()
        self.input_dim = input_dim
        
        # 预生成随机排列索引perm
        self.perm = torch.randperm(input_dim)
        
        # 预生成perm的逆排列inv_perm
        self.inv_perm = torch.argsort(self.perm)
    
    def forward(self, x):
        """
        Shuffle前向变换
        输入:
            x: 输入张量，形状(batch_size, input_dim)
        输出:
            output: 打乱后张量，形状(batch_size, input_dim)
            log_det: 雅可比行列式的对数，固定为0
        """
        # 检查输入维度
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Shuffle期望输入维度 {self.input_dim}，但得到 {x.shape[1]}")
            
        # 用perm打乱x的特征
        output = x[:, self.perm]
        
        # 返回log_det=0（因为置换矩阵的行列式为±1，对数接近0）
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        return output, log_det
    
    def inverse(self, z):
        """
        Shuffle逆向变换
        输入:
            z: 输入张量，形状(batch_size, input_dim)
        输出:
            output: 恢复顺序后张量，形状(batch_size, input_dim)
            log_det: 雅可比行列式的对数，固定为0
        """
        # 检查输入维度
        if z.shape[1] != self.input_dim:
            raise ValueError(f"Shuffle逆变换期望输入维度 {self.input_dim}，但得到 {z.shape[1]}")
            
        # 用inv_perm恢复z的特征顺序
        output = z[:, self.inv_perm]
        
        # 返回log_det=0
        log_det = torch.zeros(z.shape[0], device=z.device)
        
        return output, log_det


class RINNModel(nn.Module):
    """
    完整的 R-INN 模型：由多个 RINNBlock 顺序连接组成，RINNBlock之间加入Shuffle层
    """
    def __init__(self, input_dim, hidden_dim=64, num_blocks=3, num_stages=4, num_cycles_per_stage=2,
                 ratio_toZ_after_flowstage=0.3, ratio_x1_x2_inAffine=0.25):
        """
        初始化 RINNModel
        参数:
            input_dim: 输入特征维度
            hidden_dim: RealNVP 中 MLP 的隐藏层维度
            num_blocks: RINNBlock 的数量
            num_stages: 每个 RINNBlock 中 RealNVP 的流阶段数量
            num_cycles_per_stage: 每个流阶段中的内部循环次数
            ratio_toZ_after_flowstage: 每个流阶段后进入z输出的比例
            ratio_x1_x2_inAffine: RealNVP中Affine层x1条件部分的比例
        """
        super(RINNModel, self).__init__()
        
        # 创建交替包含RINNBlock和Shuffle层的组件列表
        components = []
        for i in range(num_blocks):
            # 添加RINNBlock
            components.append(RINNBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_stages=num_stages,
                num_cycles_per_stage=num_cycles_per_stage,
                ratio_toZ_after_flowstage=ratio_toZ_after_flowstage,
                ratio_x1_x2_inAffine=ratio_x1_x2_inAffine
            ))
            # 如果不是最后一个RINNBlock，添加Shuffle层
            # ————————————————————————————————————————该层虽文中要求有，但目前加进去效果并不理想，先这么放着——————————————————————————————————————————————————
            # if i < num_blocks - 1:
            #     components.append(Shuffle(input_dim=input_dim))
        
        # 将组件列表转换为ModuleList
        self.components = nn.ModuleList(components)
        
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        
        # 初始化FinalFeatureAdjustment层作为模型的最后一层
        self.feature_adjustment = FinalFeatureAdjustment(input_dim=input_dim)
    
    def forward(self, x, return_intermediate=False):
        """
        RINNModel 前向变换
        输入:
            x: 输入张量，形状(batch_size, input_dim)
            return_intermediate: 是否返回中间结果（RealNVP的z输出）
        输出:
            z: 变换后张量，形状(batch_size, input_dim)
            log_det_total: 雅可比行列式的对数总和，形状(batch_size,)
            (可选) z_from_realnvp: RealNVP层输出的z（用于z_loss计算）
        """
        # 检查输入维度
        if x.shape[1] != self.input_dim:
            raise ValueError(f"模型期望输入维度 {self.input_dim}，但得到 {x.shape[1]}")
        
        # 初始化 z 和 log_det_total
        z = x
        log_det_total = 0
        
        # 存储RealNVP的输出z（用于z_loss计算）
        z_from_realnvp = None
        
        # 顺序通过每个组件（RINNBlock和Shuffle层交替）
        for i, component in enumerate(self.components):
            if i == 0 and return_intermediate and hasattr(component, 'forward_with_intermediate'):
                # 使用带中间结果的forward方法
                z, log_det, z_from_realnvp = component.forward_with_intermediate(z)
                log_det_total += log_det
            else:
                z, log_det = component(z)
                log_det_total += log_det

        # 应用FinalFeatureAdjustment层进行特征精细化调整
        z = self.feature_adjustment(z)
        # 累加FinalFeatureAdjustment层的雅可比行列式对数
        log_det_adjustment = self.feature_adjustment.log_det_jacobian(z)
        log_det_total += log_det_adjustment

        if return_intermediate:
            return z, log_det_total, z_from_realnvp
        else:
            return z, log_det_total
    
    def inverse(self, z):
        """
        RINNModel 逆向变换
        输入:
            z: 输入张量，形状(batch_size, input_dim)
        输出:
            x_recon: 恢复的输入张量，形状(batch_size, input_dim)
            log_det_total: 逆过程雅可比行列式的对数总和，形状(batch_size,)
        """
        # 检查输入维度
        if z.shape[1] != self.input_dim:
            raise ValueError(f"模型逆变换期望输入维度 {self.input_dim}，但得到 {z.shape[1]}")
        
        # 初始化log_det_total
        log_det_total = 0
        
        # 首先应用FinalFeatureAdjustment层的逆向变换
        z = self.feature_adjustment.inverse(z)
        # 累加FinalFeatureAdjustment层逆变换的雅可比行列式对数（注意符号）
        log_det_adjustment = -self.feature_adjustment.log_det_jacobian(z)
        log_det_total += log_det_adjustment
        
        # 初始化 x_recon
        x_recon = z
        
        # 逆序通过每个组件（先Shuffle.inverse，再RINNBlock.inverse）
        for component in reversed(self.components):
            x_recon, log_det = component.inverse(x_recon)
            log_det_total += log_det
        
        return x_recon, log_det_total


# 测试代码：验证可逆性
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 配置参数：使用3个可逆块和20维输入输出
    input_dim = 20
    batch_size = 32
    num_blocks = 3  # 3个可逆块
    
    # 创建 RINN 模型
    model = RINNModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_blocks=num_blocks,
        num_stages=4,
        num_cycles_per_stage=2,
        ratio_toZ_after_flowstage=0.3,
        ratio_x1_x2_inAffine=0.25
    )
    
    # 生成随机输入
    x = torch.randn(batch_size, input_dim)
    
    try:
        print(f"原始输入x的形状: {x.shape}")
        print(f"使用 {num_blocks} 个可逆块的 R-INN 模型")
        
        # 前向传播
        z, log_det_forward = model(x)
        print(f"前向传播成功，z的形状: {z.shape}")
        print(f"前向log_det: {log_det_forward.mean().item():.4f}")
        
        # 逆向传播
        x_recon, log_det_inverse = model.inverse(z)
        print(f"逆向传播成功，重建x的形状: {x_recon.shape}")
        print(f"逆向log_det: {log_det_inverse.mean().item():.4f}")
        
        # 计算恢复后输入与原始输入的MSE
        mse = torch.mean((x_recon - x) ** 2)
        
        # 打印结果
        print(f"可逆性验证：MSE={mse.item():.10f}")
        if mse < 1e-5:
            print("✓ 可逆性验证通过！重建误差小于1e-5")
        else:
            print("⚠ 可逆性验证不通过，重建误差较大")
    except Exception as e:
        print(f"运行出错: {type(e).__name__}: {str(e)}")