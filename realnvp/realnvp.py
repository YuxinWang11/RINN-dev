import torch
import torch.nn as nn
import numpy as np


class GaussianPrior(nn.Module):
    """
    高斯先验类：支持固定高斯和可学习高斯
    """
    def __init__(self, dim, learnable=False):
        """
        初始化高斯先验
        参数:
            dim: 高斯分布的维度
            learnable: 是否使用可学习的均值和方差
        """
        super(GaussianPrior, self).__init__()
        self.dim = dim
        self.learnable = learnable
        
        if learnable:
            # 可学习的均值和对数方差
            self.mean = nn.Parameter(torch.zeros(dim))
            self.log_var = nn.Parameter(torch.zeros(dim))
        else:
            # 固定的单位高斯分布
            self.mean = torch.zeros(dim)
            self.log_var = torch.zeros(dim)
    
    def log_prob(self, z):
        """
        计算z的高斯对数似然
        输入:
            z: 输入张量，形状(batch_size, dim)
        输出:
            log_prob: 对数似然，形状(batch_size,)
        """
        if self.learnable:
            mean = self.mean
            log_var = self.log_var
        else:
            mean = self.mean.to(z.device)
            log_var = self.log_var.to(z.device)
        
        # 计算高斯对数似然
        # log p(z) = -0.5 * (log(2π) + log_var + (z - mean)^2 / exp(log_var))
        log_prob = -0.5 * (torch.log(torch.tensor(2 * np.pi)) + log_var + 
                         torch.pow(z - mean, 2) / torch.exp(log_var))
        return log_prob.sum(dim=1)


class AffineCoupling(nn.Module):
    """
    AffineCoupling层：将输入特征分为两部分，用一部分预测另一部分的仿射变换参数
    """
    def __init__(self, input_dim, x1_dim, hidden_dim):
        """
        初始化AffineCoupling层
        参数:
            input_dim: 输入特征维度
            x1_dim: 条件部分x1的维度
            hidden_dim: MLP隐藏层维度
        """
        super(AffineCoupling, self).__init__()
        
        # 存储输入维度和拆分信息
        self.input_dim = input_dim
        self.x1_dim = x1_dim
        self.x2_dim = input_dim - x1_dim
        
        # 确保x1_dim和x2_dim都大于0
        if self.x1_dim <= 0 or self.x2_dim <= 0:
            raise ValueError(f"AffineCoupling拆分后维度必须都大于0，但得到x1_dim={self.x1_dim}, x2_dim={self.x2_dim}")
        
        # 导入ActNorm
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from actnorm.actnorm import ActNorm1d
        
        # 用于生成缩放参数scale的MLP
        self.scale_net = nn.Sequential(
            nn.Linear(self.x1_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.x2_dim),
            nn.Tanh()  # 使用tanh限制scale范围
        )
        
        # 用于生成平移参数translate的MLP
        self.translate_net = nn.Sequential(
            nn.Linear(self.x1_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ActNorm1d(hidden_dim),
            nn.ReLU(),#尝试使用两层神经网络
            nn.Linear(hidden_dim, self.x2_dim)
        )
        
        # 耦合层输出端全局标准化
        self.output_norm = ActNorm1d(self.input_dim)
    
    def forward(self, x):
        """
        AffineCoupling前向变换
        输入:
            x: 输入张量，形状(batch_size, input_dim)
        输出:
            output: 变换后张量，形状(batch_size, input_dim)
            log_det: 该层雅可比行列式的对数，形状(batch_size,)
        """
        # 检查输入维度
        if x.shape[1] != self.input_dim:
            raise ValueError(f"AffineCoupling期望输入维度 {self.input_dim}，但得到 {x.shape[1]}")
        
        # 拆分x为条件部分和待变换部分
        h1 = x[:, :self.x1_dim]  # 条件部分
        h2 = x[:, self.x1_dim:]  # 待变换部分
        
        # 用条件部分通过MLP生成scale和translate
        scale = self.scale_net(h1)
        translate = self.translate_net(h1)
        
        # 对待变换部分应用仿射变换
        h2_out = h2 * torch.exp(scale) + translate
        
        # 合并条件部分和变换后的部分
        output = torch.cat([h1, h2_out], dim=-1)
        
        # 应用耦合层输出端全局标准化
        output = self.output_norm(output)
        
        # 计算log_det，包括output_norm的雅可比行列式
        log_det = scale.sum(dim=-1) + self.output_norm.log_det_jacobian(output)
        
        return output, log_det
    
    def inverse(self, z):
        """
        AffineCoupling逆向变换
        输入:
            z: 输入张量，形状(batch_size, input_dim)
        输出:
            output: 逆变换后张量，形状(batch_size, input_dim)
            log_det: 逆过程雅可比行列式的对数，形状(batch_size,)
        """
        # 检查输入维度
        if z.shape[1] != self.input_dim:
            raise ValueError(f"AffineCoupling逆变换期望输入维度 {self.input_dim}，但得到 {z.shape[1]}")
        
        # 应用output_norm的逆变换
        z = self.output_norm.inverse(z)
        
        # 拆分z为条件部分和变换后的部分
        z1 = z[:, :self.x1_dim]  # 条件部分
        z2 = z[:, self.x1_dim:]  # 变换后的部分
        
        # 用条件部分通过MLP生成scale和translate
        scale = self.scale_net(z1)
        translate = self.translate_net(z1)
        
        # 对变换后的部分应用逆仿射变换
        z2_out = (z2 - translate) * torch.exp(-scale)
        
        # 合并条件部分和恢复后的部分
        output = torch.cat([z1, z2_out], dim=-1)
        
        # 计算逆过程的log_det，包括output_norm逆变换的雅可比行列式
        log_det = -scale.sum(dim=-1) - self.output_norm.log_det_jacobian(output)
        
        return output, log_det


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


class FlowCell(nn.Module):
    """
    FlowCell类：包含一个AffineCoupling和一个Shuffle层
    作为FlowStage的基本构建块
    """
    def __init__(self, input_dim, x1_dim, hidden_dim):
        """
        初始化FlowCell
        参数:
            input_dim: 输入特征维度
            x1_dim: AffineCoupling中条件部分x1的维度
            hidden_dim: MLP隐藏层维度
        """
        super(FlowCell, self).__init__()
        self.affine_coupling = AffineCoupling(input_dim, x1_dim, hidden_dim)
        self.shuffle = Shuffle(input_dim)
    
    def forward(self, x):
        """
        FlowCell前向变换
        """
        x, log_det1 = self.affine_coupling(x)
        x, log_det2 = self.shuffle(x)
        return x, log_det1 + log_det2
    
    def inverse(self, z):
        """
        FlowCell逆向变换
        """
        z, log_det2 = self.shuffle.inverse(z)
        z, log_det1 = self.affine_coupling.inverse(z)
        return z, log_det1 + log_det2


class FlowStage(nn.Module):
    """
    FlowStage类：实现RINN论文中"层级拆分+内部循环"的核心逻辑
    先拆分特征，再对剩余特征执行多轮循环
    """
    def __init__(self, input_dim, z_part_dim, h_prime_dim, x1_dim, hidden_dim, num_cycles=2):
        """
        初始化FlowStage
        参数:
            input_dim: 输入特征维度
            z_part_dim: 拆分后进入z输出的维度
            h_prime_dim: 拆分后进入内部循环的维度
            x1_dim: 内部AffineCoupling中条件部分x1的维度
            hidden_dim: MLP隐藏层维度
            num_cycles: 内部循环次数
        """
        super(FlowStage, self).__init__()
        
        # 存储维度信息
        self.input_dim = input_dim
        self.z_part_dim = z_part_dim
        self.h_prime_dim = h_prime_dim
        self.num_cycles = num_cycles
        
        # 确保维度有效
        if input_dim != z_part_dim + h_prime_dim:
            raise ValueError(f"FlowStage维度不匹配：input_dim={input_dim}, z_part_dim+h_prime_dim={z_part_dim+h_prime_dim}")
        
        # 创建内部循环的FlowCell列表
        self.cells = nn.ModuleList([
            FlowCell(self.input_dim, x1_dim, hidden_dim) for _ in range(num_cycles)
        ])
    
    def forward(self, x):
        """
        FlowStage前向变换 - 先循环再拆分
        输入:
            x: 输入张量，形状(batch_size, input_dim)
        输出:
            z_part: 当前层输出z，形状(batch_size, z_part_dim)
            h_prime: 内部循环后输出，形状(batch_size, h_prime_dim)
            log_det: 该层雅可比行列式的对数总和，形状(batch_size,)
        """
        # 检查输入维度
        if x.shape[1] != self.input_dim:
            raise ValueError(f"FlowStage期望输入维度 {self.input_dim}，但得到 {x.shape[1]}")
            
        # 对整个输入x执行num_cycles次内部循环
        log_det_total = 0
        for cell in self.cells:
            x, log_det = cell(x)
            log_det_total += log_det
        
        # 拆分：前z_part_dim部分作为当前层输出z，后h_prime_dim部分作为内部循环输出h'
        z_part = x[:, :self.z_part_dim]
        h_prime = x[:, self.z_part_dim:]
        
        return z_part, h_prime, log_det_total
    
    def inverse(self, z_part, h_prime):
        """
        FlowStage逆向变换 - 先合并再循环
        输入:
            z_part: 当前层输入z，形状(batch_size, z_part_dim)
            h_prime: 内部循环输入，形状(batch_size, h_prime_dim)
        输出:
            x: 逆变换后张量，形状(batch_size, input_dim)
            log_det: 逆过程雅可比行列式的对数总和，形状(batch_size,)
        """
        # 检查输入维度
        if z_part.shape[1] != self.z_part_dim:
            raise ValueError(f"FlowStage逆变换期望z_part维度 {self.z_part_dim}，但得到 {z_part.shape[1]}")
        if h_prime.shape[1] != self.h_prime_dim:
            raise ValueError(f"FlowStage逆变换期望h_prime维度 {self.h_prime_dim}，但得到 {h_prime.shape[1]}")
            
        # 合并z_part和h_prime
        x = torch.cat([z_part, h_prime], dim=-1)
        
        # 对合并后的x逆序执行num_cycles次内部循环
        log_det_total = 0
        for cell in reversed(self.cells):
            x, log_det = cell.inverse(x)
            log_det_total += log_det
        
        return x, log_det_total


class RealNVP(nn.Module):
    """
    RealNVP主类：实现完整的归一化流模型，符合RINN论文中"层级拆分+内部循环"的设计
    """
    def __init__(self, input_dim, hidden_dim=64, num_stages=4, num_cycles_per_stage=2, 
                 ratio_toZ_after_flowstage=0.5, ratio_x1_x2_inAffine=0.5, 
                 gaussian_learnable=False):
        """
        初始化RealNVP模型
        参数:
            input_dim: 输入特征维度
            hidden_dim: FlowCell中MLP的隐藏层维度
            num_stages: 流阶段数量
            num_cycles_per_stage: 每个流阶段中的内部循环次数
            ratio_toZ_after_flowstage: FlowStage拆分后进入z输出的比例 (0,1)
            ratio_x1_x2_inAffine: AffineCoupling层中条件部分x1的比例 (0,1)
            gaussian_learnable: 是否使用可学习的高斯先验
        """
        super(RealNVP, self).__init__()
        
        # 验证参数有效性
        if not (0 < ratio_toZ_after_flowstage < 1):
            raise ValueError(f"ratio_toZ_after_flowstage必须在(0,1)之间，但得到 {ratio_toZ_after_flowstage}")
        if not (0 < ratio_x1_x2_inAffine < 1):
            raise ValueError(f"ratio_x1_x2_inAffine必须在(0,1)之间，但得到 {ratio_x1_x2_inAffine}")
        
        # 存储基本参数
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.ratio_toZ_after_flowstage = ratio_toZ_after_flowstage
        self.ratio_x1_x2_inAffine = ratio_x1_x2_inAffine
        self.gaussian_learnable = gaussian_learnable
        
        # 创建流动阶段列表
        self.stages = nn.ModuleList()
        
        # 创建用于残差链接融合的MLP层列表
        self.fusion_mlps = nn.ModuleList()
        
        # 创建高斯先验列表（每个阶段的z_part和最终的current_h都有一个高斯先验）
        self.gaussian_priors = nn.ModuleList()
        
        # 预计算并存储每个阶段的维度信息，用于前向和逆向变换
        self.stage_input_dims = []     # 每个FlowStage的输入维度
        self.z_part_dims = []          # 每个FlowStage拆分后进入z输出的维度
        self.h_prime_dims = []         # 每个FlowStage拆分后进入内部循环的维度
        self.cell_x1_dims = []         # 每个FlowCell中AffineCoupling的x1维度
        
        # 当前处理维度
        current_dim = input_dim
        
        # 预计算所有层的维度
        for i in range(num_stages):
            # 确保维度足够大以进行有效拆分
            if current_dim < 4:
                # 维度太小，使用最小有效维度
                current_dim = 4
                print(f"警告：阶段 {i+1} 维度调整为最小有效值 {current_dim}")
            
            # 计算当前阶段的维度拆分
            stage_input_dim = current_dim
            z_part_dim = max(1, int(stage_input_dim * ratio_toZ_after_flowstage))
            h_prime_dim = stage_input_dim - z_part_dim
            
            # 确保拆分后维度都大于0且h_prime_dim至少为2
            if h_prime_dim < 2:
                z_part_dim = stage_input_dim - 2
                h_prime_dim = 2
                print(f"警告：阶段 {i+1} 维度拆分调整为 z_part_dim={z_part_dim}, h_prime_dim={h_prime_dim}")
            
            # 计算FlowCell中AffineCoupling的x1维度
            x1_dim = max(1, int(h_prime_dim * ratio_x1_x2_inAffine))
            
            # 确保x1_dim有效
            if h_prime_dim - x1_dim <= 0:
                x1_dim = h_prime_dim - 1
                print(f"警告：阶段 {i+1} AffineCoupling拆分调整为 x1_dim={x1_dim}")
            
            # 存储维度信息
            self.stage_input_dims.append(stage_input_dim)
            self.z_part_dims.append(z_part_dim)
            self.h_prime_dims.append(h_prime_dim)
            self.cell_x1_dims.append(x1_dim)
            
            # 创建FlowStage
            self.stages.append(FlowStage(
                stage_input_dim, 
                z_part_dim, 
                h_prime_dim, 
                x1_dim, 
                hidden_dim, 
                num_cycles_per_stage
            ))
            
            # 创建仿射变换融合的MLP：将z_part映射到2倍h_prime的维度，分别作为scale和shift
            fusion_mlp = nn.Sequential(
                nn.Linear(z_part_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * h_prime_dim),
                nn.Tanh()  # 添加Tanh限制scale范围，确保数值稳定性
            )
            self.fusion_mlps.append(fusion_mlp)
            
            # 创建当前阶段z_part的高斯先验
            self.gaussian_priors.append(GaussianPrior(z_part_dim, learnable=gaussian_learnable))
            
            # 更新当前维度（为下一个阶段准备）
            current_dim = h_prime_dim
        
        # 为最终的current_h创建高斯先验
        self.final_gaussian_prior = GaussianPrior(current_dim, learnable=gaussian_learnable)
    
    def forward(self, x):
        """
        RealNVP整体前向变换
        输入:
            x: 输入数据，形状(batch_size, data_dim)，其中data_dim <= input_dim
        输出:
            z: 变换后的数据，形状(batch_size, latent_dim)
            log_det_total: 雅可比行列式的对数总和，形状(batch_size,)
            total_log_pz: 各阶段z_part和最终current_h的高斯对数似然之和，形状(batch_size,)
        """
        # 获取批次大小
        batch_size = x.shape[0]
        device = x.device
        
        # 确保输入维度不超过模型定义的input_dim
        if x.shape[1] > self.input_dim:
            # 截断到input_dim
            h = x[:, :self.input_dim]
            print(f"警告：输入数据维度 {x.shape[1]} 大于模型input_dim {self.input_dim}，已截断")
        elif x.shape[1] < self.input_dim:
            # 零填充到input_dim
            h = torch.zeros(batch_size, self.input_dim, device=device)
            h[:, :x.shape[1]] = x
        else:
            h = x
        
        # 初始化z_list、log_det_total和total_log_pz
        z_list = []
        log_det_total = 0
        total_log_pz = 0
        
        # 当前处理的特征
        current_h = h
        
        # 遍历每个流阶段
        for i in range(self.num_stages):
            if i >= len(self.stages):
                break
            
            # 获取当前阶段
            stage = self.stages[i]
            
            # 确保current_h维度匹配
            if current_h.shape[1] != self.stage_input_dims[i]:
                raise ValueError(f"阶段 {i+1} 期望输入维度 {self.stage_input_dims[i]}，但得到 {current_h.shape[1]}")
            
            # 执行当前阶段的前向变换（先循环再拆分）
            z_part, h_prime, log_det = stage(current_h)
            log_det_total += log_det
            
            # 计算当前z_part的高斯对数似然
            log_pz = self.gaussian_priors[i].log_prob(z_part)
            total_log_pz += log_pz
            
            # 添加当前阶段的输出z
            z_list.append(z_part)
            
            # 仿射变换融合：将z_part通过MLP映射生成scale和shift参数
            scale_shift = self.fusion_mlps[i](z_part)
            
            # 分割scale和shift
            scale = scale_shift[:, :h_prime.shape[1]]
            shift = scale_shift[:, h_prime.shape[1]:]
            
            # scale已经在fusion_mlp中通过Tanh限制在[-1, 1]范围，确保数值稳定性
            scale = scale * 0.5  # 进一步缩小scale范围到[-0.5, 0.5]，使exp(scale)更接近1
            
            # 应用仿射变换
            current_h = h_prime * torch.exp(scale) + shift  # 使用exp确保scale为正
            
            # 计算仿射变换的log雅可比矩阵：由于是元素级变换，雅可比矩阵是对角矩阵
            # 对角线元素为exp(scale)，因此log_det是scale的和
            log_det_total += scale.sum(dim=1)
        
        # 添加最后剩余的特征作为最终的z
        z_list.append(current_h)
        
        # 计算最终current_h的高斯对数似然
        log_pz_final = self.final_gaussian_prior.log_prob(current_h)
        total_log_pz += log_pz_final
        
        # 合并所有z为一个向量
        z = torch.cat(z_list, dim=-1)
        
        return z, log_det_total, total_log_pz
    
    def inverse(self, z):
        """
        RealNVP整体逆向变换
        输入:
            z: 变换后的数据，形状(batch_size, latent_dim)
        输出:
            x_recon: 恢复的输入数据，形状与原始输入相同
            log_det_total: 逆过程雅可比行列式的对数总和，形状(batch_size,)
        """
        # 初始化log_det_total
        log_det_total = 0
        
        # 获取批次大小和设备
        batch_size = z.shape[0]
        device = z.device
        
        # 拆分z为各阶段的输出部分
        z_list = []
        start_idx = 0
        
        # 拆分每个阶段的输出z_part
        for dim in self.z_part_dims:
            if start_idx + dim <= z.shape[1]:
                z_part = z[:, start_idx:start_idx + dim]
                z_list.append(z_part)
                start_idx += dim
            else:
                break
        
        # 添加最后剩余的特征
        if start_idx < z.shape[1]:
            z_list.append(z[:, start_idx:])
        
        # 从最后一个特征开始，作为初始current_h
        if z_list:
            current_h = z_list.pop()
        else:
            # 特殊情况处理
            current_h = torch.zeros(batch_size, 2, device=device, dtype=z.dtype)
        
        # 逆序遍历每个流阶段，重建原始特征
        for i in reversed(range(min(len(self.stages), len(z_list)))):
            stage = self.stages[i]
            z_part = z_list.pop()
            
            # 仿射变换融合的逆操作
            # 首先生成与前向相同的scale和shift参数
            scale_shift = self.fusion_mlps[i](z_part)
            scale = scale_shift[:, :current_h.shape[1]]
            shift = scale_shift[:, current_h.shape[1]:]
            
            # scale已经在fusion_mlp中通过Tanh限制在[-1, 1]范围，确保数值稳定性
            scale = scale * 0.5  # 进一步缩小scale范围到[-0.5, 0.5]
            
            # 应用逆仿射变换：(current_h - shift) / exp(scale)
            h_prime = (current_h - shift) * torch.exp(-scale)
            
            # 计算逆仿射变换的log雅可比矩阵：由于逆变换的雅可比矩阵是原变换雅可比矩阵的逆
            # 对角线元素为exp(-scale)，因此log_det是-scale的和
            log_det_affine = (-scale).sum(dim=1)
            
            # 调用FlowStage的inverse方法，注意顺序是z_part在前，h_prime在后
            current_h, log_det_stage = self.stages[i].inverse(z_part, h_prime)
            
            # 累加log_det
            log_det_total += log_det_affine + log_det_stage
        
        # 确保最终输出维度与原始输入维度匹配
        if current_h.shape[1] > self.input_dim:
            # 截断到原始输入维度
            x_recon = current_h[:, :self.input_dim]
        elif current_h.shape[1] < self.input_dim:
            # 填充零到原始输入维度
            padding = torch.zeros(batch_size, self.input_dim - current_h.shape[1], 
                                device=device, dtype=current_h.dtype)
            x_recon = torch.cat([current_h, padding], dim=-1)
        else:
            x_recon = current_h
        
        return x_recon, log_det_total


def calculate_loss(log_px):
    """
    计算损失函数
    输入:
        log_px: 输入数据的对数似然，形状(batch_size,)
    输出:
        loss: 负对数似然的均值，标量
    """
    return -torch.mean(log_px)


# 可逆性验证示例
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 配置参数 - 测试非偶数维度
    input_dim = 13  # 目标输入维度（非偶数）
    batch_size = 32  # 批次大小
    data_dim = 13  # 实际数据维度（等于input_dim）
    
    # 创建模型 - 使用非0.5拆分比例，可学习高斯先验
    model = RealNVP(
        input_dim=input_dim,
        hidden_dim=64,
        num_stages=4,
        num_cycles_per_stage=2,
        ratio_toZ_after_flowstage=0.3,  # 30%进入z输出
        ratio_x1_x2_inAffine=0.25,       # 25%为x1条件部分
        gaussian_learnable=True          # 使用可学习高斯先验
    )
    
    # 生成随机输入（实际数据维度等于目标输入维度）
    x = torch.randn(batch_size, data_dim)
    
    try:
        print(f"原始输入x的形状: {x.shape}")
        
        # 前向传播
        z, log_det_forward, total_log_pz = model(x)
        print(f"前向传播成功，z的形状: {z.shape}")
        print(f"前向log_det: {log_det_forward.mean().item():.4f}")
        print(f"总log_pz: {total_log_pz.mean().item():.4f}")
        
        # 计算log_px
        log_px = total_log_pz + log_det_forward
        print(f"log_px: {log_px.mean().item():.4f}")
        
        # 计算损失
        loss = calculate_loss(log_px)
        print(f"损失: {loss.item():.4f}")
        
        # 逆向传播
        x_recon, log_det_inverse = model.inverse(z)
        print(f"逆向传播成功，重建x的形状: {x_recon.shape}")
        print(f"逆向log_det: {log_det_inverse.mean().item():.4f}")
        
        # 计算恢复后输入与原始输入的MSE
        # 只比较原始输入的数据部分，不包括零填充部分
        mse = torch.mean((x_recon[:, :data_dim] - x) ** 2)
        
        # 打印结果
        print(f"可逆性验证：MSE={mse.item():.10f}")
        if mse < 1e-5:
            print("✓ 可逆性验证通过！重建误差小于1e-5")
        else:
            print("⚠ 可逆性验证不通过，重建误差较大")
    except Exception as e:
        print(f"运行出错: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()