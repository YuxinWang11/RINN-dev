import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RINNmodel import RINNModel

# 论文中的损失函数：L_total = w1*Lz + w2*Lx + w3*Ly
def calculate_loss(model, x, y, w1=1.0, w2=1.0, w3=1.0):
    """
    计算论文中定义的损失函数
    参数:
        model: RINN模型
        x: 输入特征 (batch_size, 3)
        y: 目标输出 (batch_size, 2)
        w1, w2, w3: 损失权重
    返回:
        total_loss: 总损失
        losses: 各部分损失的字典
    """
    # 填充x和y到10维
    batch_size = x.shape[0]
    x_padded = torch.zeros(batch_size, 10).to(x.device)
    x_padded[:, :3] = x
    
    y_padded = torch.zeros(batch_size, 10).to(y.device)
    y_padded[:, :2] = y
    
    # 前向传播
    z, log_det_forward = model(x_padded)
    
    # 逆向传播得到重建的x
    x_recon_padded, log_det_inverse = model.inverse(z)
    x_recon = x_recon_padded[:, :3]
    
    # 计算各部分损失
    # 1. 隐空间损失 Lz: 使用MSE衡量z与目标y的差距
    Lz = torch.mean((z - y_padded) ** 2)
    
    # 2. 输入空间损失 Lx: 使用MSE衡量重建x与原始x的差距
    Lx = torch.mean((x_recon - x) ** 2)
    
    # 3. 输出空间损失 Ly: 使用MSE衡量前向预测与真实值的差距
    # 从z中提取前2个维度作为预测的y
    y_pred = z[:, :2]
    Ly = torch.mean((y_pred - y) ** 2)
    
    # 计算总损失
    total_loss = w1 * Lz + w2 * Lx + w3 * Ly
    
    return total_loss, {"Lz": Lz, "Lx": Lx, "Ly": Ly}

# 生成符合条件的训练数据
def generate_data(batch_size=64):
    """
    生成满足条件的训练数据：
    1. x的每个元素在0~1之间
    2. x的逐个元素和为y的逐个元素和
    3. y的两个元素相等
    """
    # 生成0~1之间的随机x
    x = torch.rand(batch_size, 3)
    
    # 计算x的元素和
    x_sum = torch.sum(x, dim=1, keepdim=True)
    
    # 创建y，两个元素相等且和等于x的元素和
    y = torch.cat([x_sum / 2, x_sum / 2], dim=1)
    
    return x, y

# 训练函数
def train_model(model, num_epochs=1000, batch_size=64, lr=1e-4):
    """
    训练RINN模型
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # L2正则化
    
    # 用于记录损失
    train_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 生成训练数据
        x, y = generate_data(batch_size)
        
        # 前向传播计算损失
        total_loss, losses = calculate_loss(model, x, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录损失
        train_losses.append(total_loss.item())
        
        # 每100个epoch打印一次损失
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.6f}")
            print(f"  Lz: {losses['Lz'].item():.6f}, Lx: {losses['Lx'].item():.6f}, Ly: {losses['Ly'].item():.6f}")
    
    return train_losses

# 测试函数
def test_model(model, num_samples=100):
    """
    测试模型性能
    """
    x_test, y_test = generate_data(num_samples)
    
    # 在评估模式下进行测试
    model.eval()
    with torch.no_grad():
        total_loss, losses = calculate_loss(model, x_test, y_test)
        
        # 计算y预测
        x_padded = torch.zeros(num_samples, 10)
        x_padded[:, :3] = x_test
        z, _ = model(x_padded)
        y_pred = z[:, :2]
        
        # 计算y预测的准确率指标
        y_diff = torch.abs(y_pred - y_test)
        mean_y_error = torch.mean(y_diff).item()
        max_y_error = torch.max(y_diff).item()
        
        # 检查x重建质量
        x_recon_padded, _ = model.inverse(z)
        x_recon = x_recon_padded[:, :3]
        x_mse = torch.mean((x_recon - x_test) ** 2).item()
    
    # 回到训练模式
    model.train()
    
    print("\n测试结果:")
    print(f"总损失: {total_loss.item():.6f}")
    print(f"Lz: {losses['Lz'].item():.6f}, Lx: {losses['Lx'].item():.6f}, Ly: {losses['Ly'].item():.6f}")
    print(f"y预测平均误差: {mean_y_error:.6f}")
    print(f"y预测最大误差: {max_y_error:.6f}")
    print(f"x重建MSE: {x_mse:.6f}")
    
    # 打印一些样本
    print("\n样本结果:")
    for i in range(min(5, num_samples)):
        print(f"样本 {i+1}:")
        print(f"  原始x: {x_test[i].tolist()}")
        print(f"  重建x: {x_recon[i].tolist()}")
        print(f"  真实y: {y_test[i].tolist()}")
        print(f"  预测y: {y_pred[i].tolist()}")

# 主函数
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 创建RINN模型，输入维度为10（因为需要0填充到10维）
    model = RINNModel(
        input_dim=10,    # 填充后的维度
        hidden_dim=64,
        num_blocks=3,
        num_stages=4,
        num_cycles_per_stage=2
    )
    
    print("开始训练RINN模型...")
    print(f"模型结构: {model.__class__.__name__}")
    print(f"输入维度: {model.input_dim}")
    print(f"RINN块数量: {model.num_blocks}")
    
    # 训练模型
    train_losses = train_model(model)
    
    # 测试模型
    test_model(model)
    
    # 保存模型
    torch.save(model.state_dict(), "rinn_model.pth")
    print("\n模型已保存为 rinn_model.pth")