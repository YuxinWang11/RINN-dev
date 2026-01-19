import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import re

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建输出文件夹
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"mlp_baseline_{timestamp}"
checkpoint_dir = os.path.join('model_checkpoints_rinn', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'本次训练输出文件夹: {checkpoint_dir}')

# 设置KMP_DUPLICATE_LIB_OK环境变量，避免matplotlib冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 获取设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"使用设备: {device}")

# ============== 数据加载与预处理 ==============
print('\n=== 加载数据 ===')

# 加载s11.csv文件
data_path = 'data/s11.csv'

# 读取表头获取几何参数
with open(data_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)

# 提取几何参数信息
def extract_geometry_params(col_name):
    """从列名中提取几何参数H1, H2, H3, H_C1, H_C2"""
    col_name = col_name.replace('\n', '').replace('"', '')
    
    h1_match = re.search(r"H1='([\d.]+)mm'", col_name)
    h2_match = re.search(r"H2='([\d.]+)mm'", col_name)
    h3_match = re.search(r"H3='([\d.]+)mm'", col_name)
    hc1_match = re.search(r"H_C1='([\d.]+)mm'", col_name)
    hc2_match = re.search(r"H_C2='([\d.]+)mm'", col_name)
    
    if all([h1_match, h2_match, h3_match, hc1_match, hc2_match]):
        return [
            float(h1_match.group(1)),
            float(h2_match.group(1)),
            float(h3_match.group(1)),
            float(hc1_match.group(1)),
            float(hc2_match.group(1))
        ]
    return None

# 提取所有几何参数样本
x_samples = []
valid_columns = []

for i, col in enumerate(header[1:]):  # 跳过第一列频率
    params = extract_geometry_params(col)
    if params:
        x_samples.append(params)
        valid_columns.append(i+1)

x_features = np.array(x_samples, dtype=np.float32)
print(f'X特征形状: {x_features.shape}')
print(f'X特征示例 (第一个样本): {x_features[0]}')

# 读取S11数据
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
freq_data = data[:, 0]  # 频率数据
print(f'频率点数: {len(freq_data)}')
print(f'频率范围: {freq_data[0]} GHz - {freq_data[-1]} GHz')

# 提取有效的S11数据列
y_data = data[:, valid_columns]  # 只保留有效的列
y_data = y_data.T  # 转置为(样本数, 频率点数)
print(f'Y数据形状: {y_data.shape}')

# 数据标准化
normalization_method = 'robust'  # 'standard' 或 'robust'

# 先划分训练集和验证集的索引，再分别进行归一化
n_samples = len(x_features)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 提取训练集数据
train_x = x_features[train_indices]
train_y = y_data[train_indices]

# 提取验证集数据
val_x = x_features[val_indices]
val_y = y_data[val_indices]

# 训练集归一化
if normalization_method == 'standard':
    # 使用均值-标准差归一化
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    train_x_normalized = (train_x - x_mean) / (x_std + 1e-8)
    val_x_normalized = (val_x - x_mean) / (x_std + 1e-8)
    
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    train_y_normalized = (train_y - y_mean) / (y_std + 1e-8)
    val_y_normalized = (val_y - y_mean) / (y_std + 1e-8)
else:  # 'robust'
    # 使用四分位数归一化（中位数和四分位距）
    x_median = np.median(train_x, axis=0)
    x_q1 = np.percentile(train_x, 25, axis=0)
    x_q3 = np.percentile(train_x, 75, axis=0)
    x_iqr = x_q3 - x_q1
    train_x_normalized = (train_x - x_median) / (x_iqr + 1e-8)
    val_x_normalized = (val_x - x_median) / (x_iqr + 1e-8)
    
    y_median = np.median(train_y, axis=0)
    y_q1 = np.percentile(train_y, 25, axis=0)
    y_q3 = np.percentile(train_y, 75, axis=0)
    y_iqr = y_q3 - y_q1
    train_y_normalized = (train_y - y_median) / (y_iqr + 1e-8)
    val_y_normalized = (val_y - y_median) / (y_iqr + 1e-8)
    
    # 保存鲁棒归一化参数
    x_mean, x_std = x_median, x_iqr
    y_mean, y_std = y_median, y_iqr

print(f'归一化方法: {normalization_method}')
print(f'X特征归一化后均值: {train_x_normalized.mean(axis=0).mean():.6f}, 标准差: {train_x_normalized.std(axis=0).mean():.6f}')
print(f'Y数据归一化后均值: {train_y_normalized.mean(axis=0).mean():.6f}, 标准差: {train_y_normalized.std(axis=0).mean():.6f}')

# ============== 维度配置 ==============
print('\n=== 维度处理 ===')

# 配置参数
x_dim = x_features.shape[1]  # X维度：5
y_dim = y_data.shape[1]       # Y维度：100

print(f'X维度: {x_dim}, Y维度: {y_dim}')

# 转换为PyTorch张量
train_x_tensor = torch.FloatTensor(train_x_normalized)
train_y_tensor = torch.FloatTensor(train_y_normalized)
val_x_tensor = torch.FloatTensor(val_x_normalized)
val_y_tensor = torch.FloatTensor(val_y_normalized)

# 创建DataLoader
batch_size = 8

# X→Y的DataLoader
train_dataset_x2y = TensorDataset(train_x_tensor, train_y_tensor)
val_dataset_x2y = TensorDataset(val_x_tensor, val_y_tensor)
train_loader_x2y = DataLoader(train_dataset_x2y, batch_size=batch_size, shuffle=True)
val_loader_x2y = DataLoader(val_dataset_x2y, batch_size=batch_size, shuffle=False)

# Y→X的DataLoader
train_dataset_y2x = TensorDataset(train_y_tensor, train_x_tensor)
val_dataset_y2x = TensorDataset(val_y_tensor, val_x_tensor)
train_loader_y2x = DataLoader(train_dataset_y2x, batch_size=batch_size, shuffle=True)
val_loader_y2x = DataLoader(val_dataset_y2x, batch_size=batch_size, shuffle=False)

# ============== 模型定义 ==============
print('\n=== 模型定义 ===')

# 优化版全连接神经网络：X→Y（适度复杂度）
class X2YModel(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(X2YModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, y_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 优化版全连接神经网络：Y→X（适度复杂度）
class Y2XModel(nn.Module):
    def __init__(self, y_dim, x_dim):
        super(Y2XModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(y_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, x_dim)
        )
    
    def forward(self, y):
        return self.network(y)

# 初始化模型
model_x2y = X2YModel(x_dim, y_dim).to(device)
model_y2x = Y2XModel(y_dim, x_dim).to(device)

print(f'X→Y模型参数总数: {sum(p.numel() for p in model_x2y.parameters())}')
print(f'Y→X模型参数总数: {sum(p.numel() for p in model_y2x.parameters())}')

# ============== 训练配置 ==============
print('\n=== 训练配置 ===')

skip_training = False  # 设置为True跳过训练，False执行完整训练

# 训练参数
num_epochs = 100
lr = 1e-3
weight_decay = 1e-5
clip_value = 0.5

# 损失函数：均方误差
criterion = nn.MSELoss()

# 优化器
optimizer_x2y = optim.AdamW(model_x2y.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_y2x = optim.AdamW(model_y2x.parameters(), lr=lr, weight_decay=weight_decay)

# 学习率调度器
scheduler_x2y = optim.lr_scheduler.ReduceLROnPlateau(optimizer_x2y, mode='min', factor=0.5, patience=10, threshold=1e-6)
scheduler_y2x = optim.lr_scheduler.ReduceLROnPlateau(optimizer_y2x, mode='min', factor=0.5, patience=10, threshold=1e-6)

# ============== 训练循环：X→Y ==============
print('\n=== 开始训练X→Y模型 ===')

best_val_loss_x2y = float('inf')
patience_x2y = 20
patience_counter_x2y = 0

train_losses_x2y = []
val_losses_x2y = []

if not skip_training:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model_x2y.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in train_loader_x2y:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model_x2y(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer_x2y.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_x2y.parameters(), max_norm=clip_value)
            optimizer_x2y.step()
            
            epoch_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader_x2y)
        train_losses_x2y.append(avg_train_loss)
        
        # 验证阶段
        model_x2y.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader_x2y:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model_x2y(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / len(val_loader_x2y)
        val_losses_x2y.append(avg_val_loss)
        
        # 更新学习率
        scheduler_x2y.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss_x2y:
            best_val_loss_x2y = avg_val_loss
            patience_counter_x2y = 0
            torch.save({
                'model_state_dict': model_x2y.state_dict(),
                'x_mean': x_mean,
                'x_std': x_std,
                'y_mean': y_mean,
                'y_std': y_std
            }, os.path.join(checkpoint_dir, 'best_model_x2y.pth'))
        else:
            patience_counter_x2y += 1
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], X→Y Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # 早停
        if patience_counter_x2y >= patience_x2y:
            print(f'\nX→Y模型早停触发! 验证损失连续{patience_x2y}个epoch没有改善')
            break
else:
    print('跳过X→Y训练阶段 (skip_training=True)')
    train_losses_x2y = [1.0, 0.8, 0.6, 0.4, 0.2] * 20
    val_losses_x2y = [1.2, 0.9, 0.7, 0.5, 0.3] * 20

# ============== 训练循环：Y→X ==============
print('\n=== 开始训练Y→X模型 ===')

best_val_loss_y2x = float('inf')
patience_y2x = 20
patience_counter_y2x = 0

train_losses_y2x = []
val_losses_y2x = []

if not skip_training:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model_y2x.train()
        epoch_train_loss = 0.0
        
        for batch_y, batch_x in train_loader_y2x:
            batch_y = batch_y.to(device)
            batch_x = batch_x.to(device)
            
            # 前向传播
            outputs = model_y2x(batch_y)
            loss = criterion(outputs, batch_x)
            
            # 反向传播和优化
            optimizer_y2x.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_y2x.parameters(), max_norm=clip_value)
            optimizer_y2x.step()
            
            epoch_train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader_y2x)
        train_losses_y2x.append(avg_train_loss)
        
        # 验证阶段
        model_y2x.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_y, batch_x in val_loader_y2x:
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                
                outputs = model_y2x(batch_y)
                loss = criterion(outputs, batch_x)
                epoch_val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / len(val_loader_y2x)
        val_losses_y2x.append(avg_val_loss)
        
        # 更新学习率
        scheduler_y2x.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss_y2x:
            best_val_loss_y2x = avg_val_loss
            patience_counter_y2x = 0
            torch.save({
                'model_state_dict': model_y2x.state_dict(),
                'x_mean': x_mean,
                'x_std': x_std,
                'y_mean': y_mean,
                'y_std': y_std
            }, os.path.join(checkpoint_dir, 'best_model_y2x.pth'))
        else:
            patience_counter_y2x += 1
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Y→X Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # 早停
        if patience_counter_y2x >= patience_y2x:
            print(f'\nY→X模型早停触发! 验证损失连续{patience_y2x}个epoch没有改善')
            break
else:
    print('跳过Y→X训练阶段 (skip_training=True)')
    train_losses_y2x = [1.0, 0.8, 0.6, 0.4, 0.2] * 20
    val_losses_y2x = [1.2, 0.9, 0.7, 0.5, 0.3] * 20

# ============== 可视化训练曲线 ==============
print('\n=== 生成训练曲线 ===')

# X→Y模型损失曲线
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(train_losses_x2y, label='Train Loss', color='blue')
axes[0].plot(val_losses_x2y, label='Val Loss', color='red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('X→Y Model Training Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Y→X模型损失曲线
axes[1].plot(train_losses_y2x, label='Train Loss', color='blue')
axes[1].plot(val_losses_y2x, label='Val Loss', color='red')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Y→X Model Training Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_losses.png'), dpi=150, bbox_inches='tight')
plt.close()
print('训练曲线已保存到:', os.path.join(checkpoint_dir, 'training_losses.png'))

# ============== 固定x预测y ==============
print('\n=== 固定x预测y功能实现 ===')

# 加载最佳X→Y模型
checkpoint_x2y = torch.load(os.path.join(checkpoint_dir, 'best_model_x2y.pth'), weights_only=False)
model_x2y.load_state_dict(checkpoint_x2y['model_state_dict'])
model_x2y.eval()

# 从验证集中选取一个测试样本
test_idx = 0
x_test = val_x_tensor[test_idx].unsqueeze(0).to(device)

y_true = val_y[test_idx]  # 原始非归一化的y值

# 预测y
with torch.no_grad():
    y_pred_normalized = model_x2y(x_test)
    # 反归一化
    y_pred = y_pred_normalized.cpu().numpy().squeeze() * y_std + y_mean

print(f'固定x预测y结果:')
print(f'  真实x: {val_x[test_idx]}')
print(f'  预测y与真实y的RMSE: {np.sqrt(np.mean((y_pred - y_true)**2)):.6f}')

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(freq_data, y_true, label='True Y', color='blue')
plt.plot(freq_data, y_pred, label='Predicted Y', color='red', linestyle='--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title(f'Fixed X Prediction (Test Sample {test_idx})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(checkpoint_dir, 'fixed_x_predicted_y.png'), dpi=150, bbox_inches='tight')
plt.close()
print('固定x预测y可视化结果已保存到:', os.path.join(checkpoint_dir, 'fixed_x_predicted_y.png'))

# 保存预测结果
np.save(os.path.join(checkpoint_dir, 'fixed_x_predicted_y.npy'), y_pred)

# ============== 固定y回推x ==============
print('\n=== 固定y回推x功能实现 ===')

# 加载最佳Y→X模型
checkpoint_y2x = torch.load(os.path.join(checkpoint_dir, 'best_model_y2x.pth'), weights_only=False)
model_y2x.load_state_dict(checkpoint_y2x['model_state_dict'])
model_y2x.eval()

# 从验证集中选取一个测试样本
test_idx = 0
y_test = val_y_tensor[test_idx].unsqueeze(0).to(device)

x_true = val_x[test_idx]  # 原始非归一化的x值

# 预测x
with torch.no_grad():
    x_pred_normalized = model_y2x(y_test)
    # 反归一化
    x_pred = x_pred_normalized.cpu().numpy().squeeze() * x_std + x_mean

print(f'固定y回推x结果:')
print(f'  真实x: {x_true}')
print(f'  回推x: {x_pred}')
print(f'  预测x与真实x的RMSE: {np.sqrt(np.mean((x_pred - x_true)**2)):.6f}')

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.bar(range(x_dim), x_true, label='True X', alpha=0.6)
plt.bar(range(x_dim), x_pred, label='Predicted X', alpha=0.6)
plt.xlabel('X Feature Index')
plt.ylabel('Value')
plt.title(f'Fixed Y Backward X (Test Sample {test_idx})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(checkpoint_dir, 'fixed_y_backward_x.png'), dpi=150, bbox_inches='tight')
plt.close()
print('固定y回推x可视化结果已保存到:', os.path.join(checkpoint_dir, 'fixed_y_backward_x.png'))

# 保存预测结果
np.save(os.path.join(checkpoint_dir, 'fixed_y_backward_x.npy'), x_pred)

# ============== 保存训练配置 ==============
training_info = {
    'timestamp': timestamp,
    'model_config_x2y': {
        'model_type': 'X2YModel',
        'input_dim': x_dim,
        'output_dim': y_dim
    },
    'model_config_y2x': {
        'model_type': 'Y2XModel',
        'input_dim': y_dim,
        'output_dim': x_dim
    },
    'training_params': {
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'clip_value': clip_value,
        'num_epochs': num_epochs
    },
    'data_info': {
        'train_samples': len(train_x),
        'val_samples': len(val_x),
        'x_dim': x_dim,
        'y_dim': y_dim,
        'normalization_method': normalization_method,
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist()
    }
}

import json
with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)

print('\n=== 训练完成! ===')
print(f'模型检查点保存在: {checkpoint_dir}')
print('预测结果已保存。')