import torch
import numpy as np

# 添加项目路径以便导入模块
import sys
sys.path.append('d:\1事务\1论文\微波设计\RINN-dev\RINN-dev')

# 导入修改后的模块
from R_INN_model.loss_methods import mmd_loss
from R_INN_model.rinn_model import RINNModel

print("===== 测试 log_det_total 集成到 MMD 损失中 =====")

# 1. 测试 mmd_loss 函数的基本功能
print("\n1. 测试 mmd_loss 函数:")
# 创建测试数据
batch_size = 32
feature_dim = 6
dist1 = torch.randn(batch_size, feature_dim)
dist2 = torch.randn(batch_size, feature_dim)

# 无 log_det_total 的情况
mmd_without_logdet = mmd_loss(dist1, dist2)
print(f"  MMD 损失 (无 log_det): {mmd_without_logdet.item():.6f}")

# 有 log_det_total 的情况
log_det_total = torch.randn(batch_size)  # 模拟雅可比行列式对数
mmd_with_logdet = mmd_loss(dist1, dist2, log_det_total=log_det_total)
print(f"  MMD 损失 (有 log_det): {mmd_with_logdet.item():.6f}")

# 验证损失值差异（应该不同）
print(f"  损失差异: {mmd_with_logdet.item() - mmd_without_logdet.item():.6f}")

# 2. 测试模型是否能正常返回 log_det_total
print("\n2. 测试 RINN 模型:")
model = RINNModel(
    input_dim=6,
    hidden_dim=5,
    num_blocks=2,
    num_stages=1,
    num_cycles_per_stage=1
)

# 创建输入张量
test_input = torch.randn(4, 6)

# 前向传播
output, log_det_forward = model(test_input)
print(f"  模型输入形状: {test_input.shape}")
print(f"  模型输出形状: {output.shape}")
print(f"  log_det_forward 形状: {log_det_forward.shape}")
print(f"  log_det_forward 样本值: {log_det_forward[:2].tolist()}")

# 3. 测试梯度传播
print("\n3. 测试梯度传播:")
test_input.requires_grad = True
output, log_det_forward = model(test_input)
loss = mmd_loss(output, torch.randn_like(output), log_det_total=log_det_forward)
loss.backward()

# 检查梯度是否存在
if test_input.grad is not None:
    print(f"  ✓ 梯度传播正常工作")
    print(f"  输入梯度范数: {test_input.grad.norm().item():.6f}")
else:
    print(f"  ✗ 梯度传播失败")

print("\n===== 测试完成 =====")