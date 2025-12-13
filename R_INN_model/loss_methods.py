import torch
from typing import Optional

def mmd_loss(dist1: torch.Tensor, dist2: torch.Tensor, sigma: Optional[float] = None, log_det_total: Optional[torch.Tensor] = None, lambda_logdet: float = 0.1) -> torch.Tensor:
    """
    计算两个分布之间的最大均值差异（MMD）损失，并可选地纳入雅可比行列式对数
    
    参数:
        dist1: torch.Tensor - 第一个分布的样本张量，形状为(batch_size, feature_dim)
        dist2: torch.Tensor - 第二个分布的样本张量，形状必须与dist1相同
        sigma: Optional[float] - 高斯核的宽度参数，若为None则自动计算
        log_det_total: Optional[torch.Tensor] - 雅可比行列式对数的总和，用于正则化
        lambda_logdet: float - 雅可比行列式损失项的权重系数
    
    返回:
        torch.Tensor - 标量张量，表示组合损失值，支持反向传播
    """
    # 检查输入维度是否匹配
    assert dist1.shape == dist2.shape, "两个分布的形状必须完全匹配"
    
    # 获取批次大小和设备
    batch_size = dist1.shape[0]
    device = dist1.device
    
    # 自动计算sigma（如果为None）
    if sigma is None:
        # 合并所有样本计算两两欧氏距离
        all_samples = torch.cat([dist1, dist2], dim=0)
        
        # 计算所有样本间的欧氏距离
        # 使用广播计算点积
        xx = torch.mm(all_samples, all_samples.t())
        x2 = (all_samples ** 2).sum(1).unsqueeze(1)
        
        # 计算欧氏距离的平方：||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        dist_sq = x2 + x2.t() - 2 * xx
        
        # 排除对角线的自距离，获取所有非零距离
        non_zero_dists = dist_sq[~torch.eye(2*batch_size, dtype=bool, device=device)]
        
        # 计算距离的中位数作为sigma
        sigma = torch.sqrt(torch.median(non_zero_dists))
    
    # 计算核矩阵k11：dist1内部的核矩阵
    def compute_kernel(x, y):
        # 计算欧氏距离的平方
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        
        x_sq = torch.diag(xx).unsqueeze(1)
        y_sq = torch.diag(yy).unsqueeze(0)
        
        # 计算欧氏距离的平方
        dist_sq = x_sq + y_sq - 2 * xy
        
        # 应用高斯核函数：k(a, b) = exp(-||a-b||^2 / (2*sigma^2))
        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
        return kernel
    
    # 计算三个核矩阵
    k11 = compute_kernel(dist1, dist1)
    k22 = compute_kernel(dist2, dist2)
    k12 = compute_kernel(dist1, dist2)
    
    # 计算MMD的三个核心项
    # term1: dist1内部平均相似性（排除对角线自相关）
    if batch_size > 1:
        term1 = (k11.sum() - k11.diag().sum()) / (batch_size * (batch_size - 1))
    else:
        # 处理batch_size=1的特殊情况，避免除零
        term1 = torch.tensor(0.0, device=device)
    
    # term2: dist2内部平均相似性（排除对角线自相关）
    if batch_size > 1:
        term2 = (k22.sum() - k22.diag().sum()) / (batch_size * (batch_size - 1))
    else:
        term2 = torch.tensor(0.0, device=device)
    
    # term3: 两分布间平均相似性
    term3 = k12.sum() / (batch_size * batch_size)
    
    # 计算MMD平方值
    mmd_sq = term1 + term2 - 2 * term3
    
    # 取非负约束并开平方，添加小的epsilon避免数值不稳定
    mmd = torch.sqrt(torch.clamp(mmd_sq, min=0.0) + 1e-12)
    
    # 如果提供了log_det_total，将其纳入损失计算
    # 目标是让log_det_total接近0，确保变换接近等体积
    if log_det_total is not None:
        # 计算log_det_total的L1正则化项（鼓励变换接近等体积）
        logdet_loss = torch.abs(log_det_total).mean()
        # 组合MMD损失和log_det正则化
        combined_loss = mmd + lambda_logdet * logdet_loss
        return combined_loss
    
    return mmd

def nmse_loss(y_real: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    用于Ly的正向预测NMSE损失计算
    计算预测性能值与真实性能值之间的归一化均方误差，匹配论文中L3的定义
    
    参数:
        y_real: torch.Tensor - 真实性能值张量，形状为(batch_size, y_dim)
        y_pred: torch.Tensor - 预测性能值张量，形状必须与y_real相同
        eps: float - 数值稳定项，默认1e-8，避免分母为零
    
    返回:
        torch.Tensor - 标量张量，表示NMSE损失值，支持反向传播
    """
    # 校验输入形状一致性，确保逐样本对应
    if y_real.shape != y_pred.shape:
        raise ValueError(f"y_real和y_pred的形状必须一致，得到y_real形状: {y_real.shape}，y_pred形状: {y_pred.shape}")
    
    # 计算基础MSE（均方误差）
    mse = torch.mean((y_real - y_pred) ** 2)
    
    # 计算真实值能量（平方均值）作为归一化基准
    real_energy = torch.mean(y_real ** 2)
    
    # 计算NMSE，添加eps确保数值稳定性
    nmse = mse / (real_energy + eps)
    
    return nmse

# 示例用法（如果直接运行此文件）
if __name__ == "__main__":
    print("=== MMD损失测试 ===")
    # 示例1：输入空间Lx场景
    x_real = torch.randn(32, 6)
    x_recon = torch.randn(32, 6)
    lx_mmd = mmd_loss(dist1=x_real, dist2=x_recon, sigma=None)
    print(f"输入空间Lx的MMD损失: {lx_mmd.item():.6f}")
    
    # 示例2：潜在空间Lz场景
    z_recon = torch.randn(32, 14)
    gaussian_dist = torch.randn(32, 14)
    lz_mmd = mmd_loss(dist1=z_recon, dist2=gaussian_dist, sigma=None)
    print(f"潜在空间Lz的MMD损失: {lz_mmd.item():.6f}")
    
    print("\n=== NMSE损失测试 ===")
    # 示例：Ly场景 - 正向预测损失
    batch_size = 32
    y_dim = 20  # 论文中性能维度示例
    y_real = torch.randn(batch_size, y_dim)  # 真实性能值
    y_pred = torch.randn(batch_size, y_dim)  # 模型正向映射输出的预测性能值
    
    ly_nmse = nmse_loss(y_real=y_real, y_pred=y_pred, eps=1e-8)
    print(f"Ly（正向预测NMSE损失）: {ly_nmse.item():.6f}")
    
    # 验证反向传播功能
    print("\n=== 验证梯度传播 ===")
    y_real.requires_grad = True
    y_pred.requires_grad = True
    
    # 计算损失
    loss = nmse_loss(y_real, y_pred)
    # 反向传播
    loss.backward()
    
    # 检查梯度是否存在
    if y_pred.grad is not None:
        print("✓ 梯度传播正常工作")
    else:
        print("✗ 梯度传播失败")