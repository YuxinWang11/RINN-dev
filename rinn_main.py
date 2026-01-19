"""
RINN模型训练和验证脚本

此脚本实现了完整的RINN模型训练和验证流程，包括以下六个阶段：
1. 前期准备：导入必要的库、设备检测和参数设置
2. 数据生成与预处理：生成符合规则的训练和验证数据
   - x是3维向量，y是6维向量，z是3维向量
   - x的取值范围为0~1之间的随机数
   - y的所有元素都等于x的元素和
   - z从标准高斯分布采样
   - 生成约30000个样本，并按8:2比例划分为训练集和验证集
3. 组件初始化：创建模型、优化器和损失函数
4. 模型训练：执行模型训练循环，包含多损失计算（Lx、Ly、Lz）
5. 模型验证：在验证集上评估模型性能
6. 核心能力验证：测试模型的正向预测和多解生成能力

数据规则：
- x是3维向量，每个元素为0~1之间的随机数
- y是6维向量，所有元素都等于x的元素和
- z是3维向量，从标准高斯分布采样
- 总样本数约30000个，按8:2比例划分训练集和验证集
"""
import os
# 设置环境变量以解决OpenMP运行时库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import matplotlib.pyplot as plt
# 设置matplotlib中文字体

# 添加项目路径
sys.path.append('c:/Users/GoODCaT/Desktop/papers/RINN')

# 导入自定义模块
from R_INN_model.device_utils import get_device
from R_INN_model.loss_methods import mmd_loss, nmse_loss
from R_INN_model.rinn_model import RINNModel
from torch.utils.data import TensorDataset, DataLoader

# 确保使用GPU（如果可用）
torch.backends.cudnn.benchmark = True  # 为卷积网络优化CUDA
if torch.cuda.is_available():
    print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    torch.cuda.empty_cache()  # 释放未使用的缓存

# 阶段一：前期准备
print("===== 阶段一：前期准备 =====")

# 配置参数
batch_size = 64  # 增大批次大小以加速训练
epochs = 200  # 增加训练轮数以充分训练模型
lr = 1.0e-3  # 降低学习率以稳定训练
weight_decay = 1e-5  # 添加权重衰减防止过拟合----
w_x = 1.0  # 调整权重系数使其接近
w_y = 2.0  # 调整权重系数使其接近
w_z = 1.0  # 调整权重系数使其接近
clip_value = 1.0  # 梯度裁剪阈值----
x_dim = 3
z_dim = 3
y_dim = 6
model_input_dim = x_dim + z_dim

# 获取计算设备
device = get_device()
print(f"使用设备: {device}")

# 初始化模型
model = RINNModel(
    input_dim=model_input_dim,
    hidden_dim=15,
    num_blocks=8,
    num_stages=1,
    num_cycles_per_stage=1
).to(device)

print(f"模型已创建，输入输出维度: {model.input_dim}")

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 添加权重衰减------


# 阶段二：数据生成与预处理
print("\n===== 阶段二：数据生成与预处理 =====")

# 生成数据集
print("生成数据集...")
# 总样本数30000个
n_samples_total = 4000

# 按照8:2的比例划分训练集和验证集
train_ratio = 0.8
n_samples_train = int(n_samples_total * train_ratio)
n_samples_val = n_samples_total - n_samples_train

print(f"总样本数: {n_samples_total}")
print(f"训练集样本数: {n_samples_train}")
print(f"验证集样本数: {n_samples_val}")

# 生成训练数据
print("生成训练数据...")
# x的取值范围为0~1之间的随机数
train_x = np.random.rand(n_samples_train, x_dim)
# 计算每个样本的元素和
train_x_sum = np.sum(train_x, axis=1, keepdims=True)
# 生成y样本，所有元素都等于x的元素和
train_y = np.repeat(train_x_sum, y_dim, axis=1)
# 生成z样本，从标准高斯分布采样
train_z = np.random.randn(n_samples_train, z_dim)

# 生成验证数据
print("生成验证数据...")
val_x = np.random.rand(n_samples_val, x_dim)
val_x_sum = np.sum(val_x, axis=1, keepdims=True)
val_y = np.repeat(val_x_sum, y_dim, axis=1)
# 生成z样本，从标准高斯分布采样
val_z = np.random.randn(n_samples_val, z_dim)

# 转换为张量并移至设备
train_x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
train_z_tensor = torch.tensor(train_z, dtype=torch.float32).to(device)

val_x_tensor = torch.tensor(val_x, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32).to(device)
val_z_tensor = torch.tensor(val_z, dtype=torch.float32).to(device)

# 数据封装
train_dataset = TensorDataset(train_x_tensor, train_y_tensor, train_z_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_x_tensor, val_y_tensor, val_z_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {n_samples_train}，验证集大小: {n_samples_val}")

# 阶段三：组件初始化
print("\n===== 阶段三：组件初始化 =====")



# 定义总损失计算逻辑
def calculate_total_loss(x_real, x_recon, y_real, y_pred, z_real, z_recon, log_det_total=None):
    # Lx: 比较真实x与重建x的分布差异，并纳入雅可比行列式对数
    Lx = mmd_loss(x_real, x_recon, log_det_total=log_det_total)
    
    # Ly: 计算真实y与预测y的正向预测误差
    # 注意：这里使用nmse_loss作为论文中提到的nmse_loss_for_ly
    Ly = nmse_loss(y_real, y_pred)
    
    # Lz: 比较重建z与标准高斯分布的差异
    Lz = mmd_loss(z_recon, z_real)
    
    # 总损失 = w_x×Lx + w_y×Ly + w_z×Lz
    total_loss = w_x * Lx + w_y * Ly + w_z * Lz
    
    return {
        "total_loss": total_loss,
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz
    }

# 阶段四：模型训练
print("\n===== 阶段四：模型训练 =====")

best_val_loss = float('inf')
save_dir = "./model_checkpoints"
os.makedirs(save_dir, exist_ok=True)

total_train_time = 0.0

# 初始化损失历史记录
loss_history = {
    'train_total': [],
    'train_Lx': [],
    'train_Ly': [],
    'train_Lz': [],
    'val_total': [],
    'val_Lx': [],
    'val_Ly': [],
    'val_Lz': []
}

for epoch in range(epochs):
    # 开始时间记录
    epoch_start_time = time.time()
    
    # 切换模型为训练模式
    model.train()
    epoch_train_losses = {"total_loss": 0.0, "Lx": 0.0, "Ly": 0.0, "Lz": 0.0}
    
    # 遍历训练集批量数据
    for batch_x, batch_y, batch_z in train_loader:
        # 正向映射
        # 拼接x和z为20维输入
        model_input = torch.cat([batch_x, batch_z], dim=1)
        # 传入模型得到20维输出
        model_output, log_det_forward = model(model_input)
        # 前y_dim维作为预测的y
        y_pred = model_output[:, :y_dim]
        
        # 反向映射
        xz_recon, log_det_inverse = model.inverse(batch_y)
        # 拆分重建的x和z
        x_recon = xz_recon[:, :x_dim]
        z_recon = xz_recon[:, x_dim:]
        
        # 计算损失，使用前向传播的log_det_total
        losses = calculate_total_loss(batch_x, x_recon, batch_y, y_pred, batch_z, z_recon, log_det_total=log_det_forward)
        
        # 梯度更新
        optimizer.zero_grad()
        losses["total_loss"].backward()
        # 梯度裁剪，防止梯度爆炸导致loss波动-----
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        
        # 损失累加
        for key in epoch_train_losses:
            epoch_train_losses[key] += losses[key].item()
    
    # 计算本轮平均损失
    num_train_batches = len(train_loader)
    for key in epoch_train_losses:
        epoch_train_losses[key] /= num_train_batches
    
    # 打印训练日志
    epoch_train_time = time.time() - epoch_start_time
    total_train_time += epoch_train_time
    
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {epoch_train_losses['total_loss']:.6f}, "
          f"Lx: {epoch_train_losses['Lx']:.6f}, "
          f"Ly: {epoch_train_losses['Ly']:.6f}, "
          f"Lz: {epoch_train_losses['Lz']:.6f}, "
          f"Train Time: {epoch_train_time:.2f}s")
    
    # 阶段五：模型验证
    val_start_time = time.time()
    
    # 切换模型为评估模式
    model.eval()
    epoch_val_losses = {"total_loss": 0.0, "Lx": 0.0, "Ly": 0.0, "Lz": 0.0}
    
    with torch.no_grad():
        for batch_x, batch_y, batch_z in val_loader:
            # 正向映射
            model_input = torch.cat([batch_x, batch_z], dim=1)
            model_output, log_det_forward = model(model_input)
            y_pred = model_output[:, :y_dim]
            
            # 反向映射 - 使用真实的batch_y而不是model_output
            xz_recon, log_det_inverse = model.inverse(batch_y)
            x_recon = xz_recon[:, :x_dim]
            z_recon = xz_recon[:, x_dim:]
            
            # 计算损失，在验证时也使用log_det_total以保持一致性
            losses = calculate_total_loss(batch_x, x_recon, batch_y, y_pred, batch_z, z_recon, log_det_total=log_det_forward)
            
            # 损失累加
            for key in epoch_val_losses:
                epoch_val_losses[key] += losses[key].item()
    
    # 计算验证时间
    val_time = time.time() - val_start_time
    
    # 计算验证集平均损失
    num_val_batches = len(val_loader)
    for key in epoch_val_losses:
        epoch_val_losses[key] /= num_val_batches
    
    # 打印验证日志
    print(f"          Val Loss: {epoch_val_losses['total_loss']:.6f}, "
          f"Lx: {epoch_val_losses['Lx']:.6f}, "
          f"Ly: {epoch_val_losses['Ly']:.6f}, "
          f"Lz: {epoch_val_losses['Lz']:.6f}, "
          f"Val Time: {val_time:.2f}s")
    
    # 记录损失历史
    loss_history['train_total'].append(epoch_train_losses['total_loss'])
    loss_history['train_Lx'].append(epoch_train_losses['Lx'])
    loss_history['train_Ly'].append(epoch_train_losses['Ly'])
    loss_history['train_Lz'].append(epoch_train_losses['Lz'])
    loss_history['val_total'].append(epoch_val_losses['total_loss'])
    loss_history['val_Lx'].append(epoch_val_losses['Lx'])
    loss_history['val_Ly'].append(epoch_val_losses['Ly'])
    loss_history['val_Lz'].append(epoch_val_losses['Lz'])
    
    # 保存最优模型
    if epoch_val_losses['total_loss'] < best_val_loss:
        best_val_loss = epoch_val_losses['total_loss']
        model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  已保存最优模型，验证损失: {best_val_loss:.6f}")

print(f"\n总训练时间: {total_train_time:.2f}s")
print(f"平均每轮训练时间: {total_train_time/epochs:.2f}s")

# 绘制损失曲线函数
def plot_loss_curves(history, save_dir=None):
    """
    Plot training and validation loss curves (excluding the first 10 epochs)
    
    Parameters:
    history: Dictionary containing training and validation loss history
    save_dir: Directory to save the figure, if None, figure won't be saved
    """
    # Exclude the first 10 epochs, start from the 11th epoch
    start_epoch = 10  # Start from epoch 11 (index 10)
    epochs = range(start_epoch + 1, len(history['train_total']) + 1)
    
    # Get data starting from start_epoch
    train_total = history['train_total'][start_epoch:]
    val_total = history['val_total'][start_epoch:]
    train_Lx = history['train_Lx'][start_epoch:]
    val_Lx = history['val_Lx'][start_epoch:]
    train_Ly = history['train_Ly'][start_epoch:]
    val_Ly = history['val_Ly'][start_epoch:]
    train_Lz = history['train_Lz'][start_epoch:]
    val_Lz = history['val_Lz'][start_epoch:]
    
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot total loss curve
    axs[0, 0].plot(epochs, train_total, 'b-', label='Train Total Loss')
    axs[0, 0].plot(epochs, val_total, 'r-', label='Val Total Loss')
    axs[0, 0].set_title('Total Loss Trend (Starting from Epoch 11)')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Lx loss curve
    axs[0, 1].plot(epochs, train_Lx, 'b-', label='Train Lx')
    axs[0, 1].plot(epochs, val_Lx, 'r-', label='Val Lx')
    axs[0, 1].set_title('Lx Loss Trend (Starting from Epoch 11)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Lx Loss Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Ly loss curve
    axs[1, 0].plot(epochs, train_Ly, 'b-', label='Train Ly')
    axs[1, 0].plot(epochs, val_Ly, 'r-', label='Val Ly')
    axs[1, 0].set_title('Ly Loss Trend (Starting from Epoch 11)')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Ly Loss Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Lz loss curve
    axs[1, 1].plot(epochs, train_Lz, 'b-', label='Train Lz')
    axs[1, 1].plot(epochs, val_Lz, 'r-', label='Val Lz')
    axs[1, 1].set_title('Lz Loss Trend (Starting from Epoch 11)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Lz Loss Value')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # If save directory is provided, save the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves figure saved to: {save_path}")
    
    # Display the figure
    plt.show()

# 创建一个单独的图像显示所有损失曲线在一个图上
def plot_all_losses(history, save_dir=None):
    """
    在一个图上绘制所有损失曲线（去除前10个epoch的数据）
    
    参数:
    history: 包含训练和验证损失历史的字典
    save_dir: 保存图片的目录，如果为None则不保存
    """
    # 去除前10个epoch的数据，从第11个epoch开始显示
    start_epoch = 10  # 从第11个epoch开始（索引为10）
    epochs = range(start_epoch + 1, len(history['train_total']) + 1)
    
    # 获取从start_epoch开始的数据
    train_total = history['train_total'][start_epoch:]
    val_total = history['val_total'][start_epoch:]
    train_Lx = history['train_Lx'][start_epoch:]
    val_Lx = history['val_Lx'][start_epoch:]
    train_Ly = history['train_Ly'][start_epoch:]
    val_Ly = history['val_Ly'][start_epoch:]
    train_Lz = history['train_Lz'][start_epoch:]
    val_Lz = history['val_Lz'][start_epoch:]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制所有训练损失曲线
    plt.plot(epochs, train_total, 'b-', label='训练总损失')
    plt.plot(epochs, train_Lx, 'g-', label='训练Lx')
    plt.plot(epochs, train_Ly, 'r-', label='训练Ly')
    plt.plot(epochs, train_Lz, 'c-', label='训练Lz')
    
    # 绘制所有验证损失曲线（使用虚线样式）
    plt.plot(epochs, val_total, 'b--', label='验证总损失')
    plt.plot(epochs, val_Lx, 'g--', label='验证Lx')
    plt.plot(epochs, val_Ly, 'r--', label='验证Ly')
    plt.plot(epochs, val_Lz, 'c--', label='验证Lz')
    
    plt.title('Comparison of All Loss Curves (Starting from Epoch 11)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # If save directory is provided, save the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'all_losses_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All losses comparison figure saved to: {save_path}")
    
    # 显示图片
    plt.show()

# 阶段四：损失可视化
print("\n===== 损失可视化 =====")
# 绘制并保存损失曲线
plot_loss_curves(loss_history, save_dir)
plot_all_losses(loss_history, save_dir)

# 阶段六：核心能力验证（多解生成能力）
print("\n===== 阶段六：核心能力验证 =====")

# 加载最优模型
model.load_state_dict(torch.load(model_path))
model.eval()

# 1. 使用5个不同的x来预测y
print("\n1. 使用5个不同的x来预测y：")

# 生成5个不同的x样本，现在取值范围为0~1之间的随机数
test_x_samples = np.array([
    [0.1, 0.2, 0.3],  # 元素和为0.6
    [0.2, 0.2, 0.2],  # 元素和为0.6
    [0.5, 0.5, 0.5],  # 元素和为1.5
    [0.0, 0.0, 1.0],  # 元素和为1.0
    [0.9, 0.8, 0.7]   # 元素和为2.4
], dtype=np.float32)

# 生成对应的z样本，从标准高斯分布采样
test_z_samples = np.random.randn(test_x_samples.shape[0], z_dim)

# 转换为张量并移至设备
test_x_tensor = torch.tensor(test_x_samples, dtype=torch.float32).to(device)
test_z_tensor = torch.tensor(test_z_samples, dtype=torch.float32).to(device)

# 预测y值
predicted_y_list = []

with torch.no_grad():
    for i in range(test_x_samples.shape[0]):
        # 拼接x和z为输入
        model_input = torch.cat([test_x_tensor[i:i+1], test_z_tensor[i:i+1]], dim=1)
        
        # 正向映射得到y预测
        model_output, _ = model(model_input)
        y_pred = model_output[:, :y_dim].cpu().numpy()[0]
        predicted_y_list.append(y_pred)
        
        # 计算x的元素和
        x_sum = np.sum(test_x_samples[i])
        
        # 打印结果
        print(f"\n测试样本 {i+1}:")
        print(f"x = {test_x_samples[i]}")
        print(f"x的元素和 = {x_sum:.2f}")
        print(f"预测的y = {y_pred}")
        print(f"y预测与x和的误差: {np.abs(y_pred[0] - x_sum):.6f}")

# 2. 生成3个合理的y作为例子，并展示多解生成能力
print("\n\n2. 生成3个合理的y并展示多解生成能力：")

# 生成3个合理的y值（每个元素都相等，范围在0~3之间，因为x是0~1之间的随机数，3个维度）
test_y_values = [0.5, 1.5, 2.5]  # 选择3个合理的值，对应x元素和的不同范围

for y_value_idx, target_y_sum in enumerate(test_y_values):
    print(f"\n===== 处理y值: {target_y_sum} =====")
    
    # 创建y向量，所有元素都等于target_y_sum
    y_vector = np.full((1, y_dim), target_y_sum, dtype=np.float32)
    y_tensor = torch.tensor(y_vector, dtype=torch.float32).to(device)
    
    # 多解生成：为同一个y生成多个不同的x和z
    n_multisolutions = 3  # 为每个y生成3个解
    generated_x_list = []
    generated_z_list = []
    
    print(f"为y={target_y_sum}生成3个解（包括x和z）：")
    
    with torch.no_grad():
        for i in range(n_multisolutions):
            # 为了生成多解，在y向量上添加微小的随机扰动
            # 这样可以在保持y值大致不变的情况下探索不同的x解
            y_perturbed = y_tensor + torch.normal(mean=0, std=0.1, size=y_tensor.shape).to(device)
            
            # 执行逆映射，使用扰动后的y作为输入
            xz_recon, _ = model.inverse(y_perturbed)
            x_recon = xz_recon[:, :x_dim].cpu().numpy()[0]
            z_recon = xz_recon[:, x_dim:].cpu().numpy()[0]
            
            # 将x值调整到0~1的范围内
            adjusted_x = np.clip(x_recon, 0, 1)
            
            generated_x_list.append(adjusted_x)
            generated_z_list.append(z_recon)
            
            # 计算调整后x的元素和
            adjusted_x_sum = np.sum(adjusted_x)
            
            # 打印生成的x、z和解的质量
            print(f"\n生成的解 {i+1}:")
            print(f"x = {adjusted_x}")
            print(f"z = {z_recon}")
            print(f"x的元素和 = {adjusted_x_sum:.2f}")
            print(f"目标y值 = {target_y_sum:.2f}")
            print(f"误差 = {np.abs(adjusted_x_sum - target_y_sum):.6f}")

# 核心能力验证总结
print("\n核心能力验证总结:")
print(f"1. 样本数量: 总样本数{30000}个，按照8:2的比例划分训练集和验证集")
print(f"2. 数据维度: x为{x_dim}维，y为{y_dim}维，z为{z_dim}维")
print(f"4. y值范围: 由于x有{x_dim}个维度且每个维度都在0~1范围内，y的取值范围是0~{x_dim}")
print(f"7. 多解生成: 为3个不同的y值（{test_y_values}）分别生成3个解，同时输出x和z")


print("\n训练和验证过程完成！")