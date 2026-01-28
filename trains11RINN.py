"""
基于RINN模型的密度估计实现
正确的数据结构和损失计算：
1. 左侧输入：X(5维) + 零填充(100维) → 总105维
2. 右侧输入：Y(100维) + Z(5维) → 总105维，其中Z是随机生成的标准高斯分布
3. 损失计算：
   - 正向预测的Y'和真实Y的NMSE损失
   - 重建X的损失
   - 正向预测的Z'和标准高斯分布的MMD差异
"""
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import re
import argparse
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ============== 参数解析 ==============
def parse_args():
    parser = argparse.ArgumentParser(description='RINN模型训练脚本')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    return parser.parse_args()

args = parse_args()

# 加载配置
config = {
    "model_config": {
        "hidden_dim": 56,
        "num_blocks": 5,
        "num_stages": 2,
        "num_cycles_per_stage": 2,
        "ratio_toZ_after_flowstage": 0.3,
        "ratio_x1_x2_inAffine": None  # 会在后面计算
    },
    "training_params": {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0005,
        "weight_decay": 1e-05,
        "clip_value": 0.5,
        "num_epochs": 150,
        "loss_weights": {
            "weight_y": 1.5,
            "weight_x": 0.5,
            "weight_z": 0.3
        }
    },
    "data_params": {
        "normalization_method": "robust"
    }
}

# 如果提供了配置文件，加载配置
if args.config and os.path.exists(args.config):
    print(f'从配置文件加载参数: {args.config}')
    with open(args.config, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
    # 更新配置
    if 'model_config' in loaded_config:
        config['model_config'].update(loaded_config['model_config'])
    if 'training_params' in loaded_config:
        config['training_params'].update(loaded_config['training_params'])
    if 'data_params' in loaded_config:
        config['data_params'].update(loaded_config['data_params'])
    print('配置加载完成!')

# ============== 创建训练输出文件夹 ==============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"rinn_correct_structure_{timestamp}"
checkpoint_dir = os.path.join('model_checkpoints_rinn', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'本次训练输出文件夹: {checkpoint_dir}')

# 保存使用的配置
with open(os.path.join(checkpoint_dir, 'used_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设备配置
from R_INN_model.device_utils import get_device
device = get_device()

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
normalization_method = config['data_params']['normalization_method']  # 'standard' 或 'robust'

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
    
    # 对Y数据进行更鲁棒的处理，先裁剪异常值
    y_median = np.median(train_y, axis=0)
    y_q1 = np.percentile(train_y, 25, axis=0)
    y_q3 = np.percentile(train_y, 75, axis=0)
    y_iqr = y_q3 - y_q1
    
    # 裁剪异常值到[Q1-3*IQR, Q3+3*IQR]范围内
    y_lower_bound = y_q1 - 3 * y_iqr
    y_upper_bound = y_q3 + 3 * y_iqr
    
    # 对训练数据进行裁剪
    train_y_clipped = np.clip(train_y, y_lower_bound, y_upper_bound)
    
    # 使用裁剪后的数据重新计算归一化参数
    y_median_clipped = np.median(train_y_clipped, axis=0)
    y_q1_clipped = np.percentile(train_y_clipped, 25, axis=0)
    y_q3_clipped = np.percentile(train_y_clipped, 75, axis=0)
    y_iqr_clipped = y_q3_clipped - y_q1_clipped
    
    # 归一化
    train_y_normalized = (train_y_clipped - y_median_clipped) / (y_iqr_clipped + 1e-8)
    # 对验证数据也进行同样的裁剪和归一化
    val_y_clipped = np.clip(val_y, y_lower_bound, y_upper_bound)
    val_y_normalized = (val_y_clipped - y_median_clipped) / (y_iqr_clipped + 1e-8)
    
    # 保存鲁棒归一化参数
    x_mean, x_std = x_median, x_iqr
    y_mean, y_std = y_median_clipped, y_iqr_clipped

# 合并归一化后的训练集和验证集
x_features_normalized = np.zeros_like(x_features, dtype=np.float32)
x_features_normalized[train_indices] = train_x_normalized
x_features_normalized[val_indices] = val_x_normalized

y_features_normalized = np.zeros_like(y_data, dtype=np.float32)
y_features_normalized[train_indices] = train_y_normalized
y_features_normalized[val_indices] = val_y_normalized

# 数据质量检查
print(f'归一化方法: {normalization_method}')
print(f'X特征归一化后均值: {x_features_normalized.mean(axis=0).mean():.6f}, 标准差: {x_features_normalized.std(axis=0).mean():.6f}')
print(f'Y数据归一化后均值: {y_features_normalized.mean(axis=0).mean():.6f}, 标准差: {y_features_normalized.std(axis=0).mean():.6f}')

# 检查是否存在NaN或无穷大值
print(f'X特征是否包含NaN: {np.isnan(x_features_normalized).any()}')
print(f'X特征是否包含无穷大: {np.isinf(x_features_normalized).any()}')
print(f'Y数据是否包含NaN: {np.isnan(y_features_normalized).any()}')
print(f'Y数据是否包含无穷大: {np.isinf(y_features_normalized).any()}')

# 检查归一化后的数据范围
print(f'X特征归一化后最小值: {x_features_normalized.min():.6f}, 最大值: {x_features_normalized.max():.6f}')
print(f'Y数据归一化后最小值: {y_features_normalized.min():.6f}, 最大值: {y_features_normalized.max():.6f}')

# ============== 维度处理：正确的数据结构 ==============
print('\n=== 维度处理与数据结构 ===')

# 配置参数
x_dim = x_features.shape[1]  # X维度：5
y_dim = y_data.shape[1]       # Y维度：100
z_dim = x_dim                 # Z维度：5（与X维度相同）

# 左侧输入：X + 零填充 → 总维度 = x_dim + padding_dim = x_dim + y_dim = 105
padding_dim = y_dim
left_input_dim = x_dim + padding_dim

# 右侧输入：Y + Z → 总维度 = y_dim + z_dim = 105
right_input_dim = y_dim + z_dim

print(f'X维度: {x_dim}, Y维度: {y_dim}, Z维度: {z_dim}')
print(f'左侧输入维度: {left_input_dim} (X: {x_dim} + 零填充: {padding_dim})')
print(f'右侧输入维度: {right_input_dim} (Y: {y_dim} + Z: {z_dim})')
print(f'总输入/输出维度: {left_input_dim} (左右侧维度相同)')

# 配置affine coupling比率，根据X和Y的维度调整
if config['model_config']['ratio_x1_x2_inAffine'] is None:
    ratio_x1_x2_inAffine = x_dim / left_input_dim  # X部分作为条件输入的比例
else:
    ratio_x1_x2_inAffine = config['model_config']['ratio_x1_x2_inAffine']
ratio_toZ_after_flowstage = config['model_config']['ratio_toZ_after_flowstage']

print(f'Affine coupling比率: {ratio_x1_x2_inAffine}')

# 创建训练集数据
# 左侧输入：X + 零填充
left_train_input = np.concatenate((train_x_normalized, np.zeros((train_size, padding_dim), dtype=np.float32)), axis=1)
left_val_input = np.concatenate((val_x_normalized, np.zeros((n_samples - train_size, padding_dim), dtype=np.float32)), axis=1)

# 右侧输入：Y + Z（Z是随机生成的标准高斯分布）
train_z = np.random.randn(train_size, z_dim).astype(np.float32)
val_z = np.random.randn(n_samples - train_size, z_dim).astype(np.float32)

right_train_input = np.concatenate((train_y_normalized, train_z), axis=1)
right_val_input = np.concatenate((val_y_normalized, val_z), axis=1)

# 转换为torch张量
left_train = torch.FloatTensor(left_train_input)
right_train = torch.FloatTensor(right_train_input)
left_val = torch.FloatTensor(left_val_input)
right_val = torch.FloatTensor(right_val_input)

print('\n数据集划分:')
print(f'  训练集: {len(left_train)} 样本')
print(f'  验证集: {len(left_val)} 样本')

# 创建DataLoader
batch_size = config['training_params']['batch_size']  # 从配置中获取batch_size，提高训练稳定性
train_dataset = TensorDataset(left_train, right_train)
val_dataset = TensorDataset(left_val, right_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 从配置中获取batch_size，提高验证效率

# ============== 模型定义 ==============
print('\n=== 模型定义 ===')

# 导入RINN模型组件
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss

# 创建可逆模型（密度估计任务）
model = RINNModel(
    input_dim=left_input_dim,  # 模型输入/输出维度：左右侧维度相同
    hidden_dim=config['model_config']['hidden_dim'],  # 从配置中获取hidden_dim，提高模型拟合能力
    num_blocks=config['model_config']['num_blocks'],   # 从配置中获取num_blocks，增强模型表达能力
    num_stages=config['model_config']['num_stages'],   # 从配置中获取num_stages，控制模型深度
    num_cycles_per_stage=config['model_config']['num_cycles_per_stage'],  # 从配置中获取num_cycles_per_stage
    ratio_toZ_after_flowstage=ratio_toZ_after_flowstage,
    ratio_x1_x2_inAffine=ratio_x1_x2_inAffine  # 使用适合X的affine coupling比率
).to(device)

print(f'模型输入/输出维度: {model.input_dim}')
print(f'左侧输入: X({x_dim}) + 零填充({padding_dim})')
print(f'右侧输入: Y({y_dim}) + Z({z_dim})')

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数总数: {total_params}')

# ============== 训练配置 ==============
print('\n=== 训练配置 ===')

# 调试标志：启用训练模式
skip_training = False  # 设置为True跳过训练，False执行完整训练

# 损失权重
weight_y = config['training_params']['loss_weights']['weight_y']  # 从配置中获取weight_y，提高正向预测精度
weight_x = config['training_params']['loss_weights']['weight_x']  # 从配置中获取weight_x，保持X重建损失权重不变
weight_z = config['training_params']['loss_weights']['weight_z']  # 从配置中获取weight_z，平衡生成多样性和稳定性
clip_value = config['training_params']['clip_value']  # 从配置中获取clip_value，防止梯度爆炸

# 优化器
lr = config['training_params']['learning_rate']  # 从配置中获取学习率，使用更精细的学习率调整
weight_decay = config['training_params']['weight_decay']  # 从配置中获取权重衰减，平衡正则化
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用AdamW优化器

# 使用学习率衰减策略
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6)  # 调整patience，平衡收敛速度和稳定性

# 训练参数
num_epochs = config['training_params']['num_epochs']  # 从配置中获取训练轮数，让模型有更多时间拟合
best_val_loss = float('inf')
patience = 60  # 增加早停耐心，避免过早停止
patience_counter = 0

# 梯度累积步数
grad_accum_steps = config['training_params']['gradient_accumulation_steps']  # 从配置中获取梯度累积步数

# 训练历史
train_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}
val_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}

# 保存训练配置信息
training_info = {
    'timestamp': timestamp,
    'model_config': {
        'input_dim': model.input_dim,
        'hidden_dim': 56,
        'num_blocks': 5,
        'num_stages': 2,
        'num_cycles_per_stage': 2,
        'ratio_toZ_after_flowstage': ratio_toZ_after_flowstage,
        'ratio_x1_x2_inAffine': ratio_x1_x2_inAffine
    },
    'training_params': {
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_accum_steps,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'clip_value': clip_value,
        'num_epochs': num_epochs,
        'loss_weights': {
            'weight_y': weight_y,
            'weight_x': weight_x,
            'weight_z': weight_z
        }
    },
    'data_info': {
        'train_samples': len(left_train),
        'val_samples': len(left_val),
        'x_dim': x_dim,
        'y_dim': y_dim,
        'z_dim': z_dim,
        'left_input_dim': left_input_dim,
        'right_input_dim': right_input_dim,
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

# ============== 损失计算函数 ==============
def calculate_loss(left_input, right_input):
    """计算正确的RINN损失
    核心：
    1. 正向预测的Y'和真实Y的加权NMSE损失
    2. 重建X的损失
    3. 正向预测的Z'和标准高斯分布的MMD差异
    """
    # 正向映射：left_input → predicted_right
    predicted_right, log_det_forward, _ = model(left_input, return_intermediate=True)
    
    # 从predicted_right中提取Y'和Z'
    predicted_y = predicted_right[:, :y_dim]
    predicted_z = predicted_right[:, y_dim:]
    
    # 从right_input中提取真实Y
    real_y = right_input[:, :y_dim]
    
    # 1. Y预测损失：加权NMSE
    y_loss = weighted_nmse_loss(real_y, predicted_y)
    
    # 2. X重建损失：从predicted_right重建X
    # 反向映射：predicted_right → reconstructed_left
    reconstructed_left, log_det_backward = model.inverse(predicted_right)
    
    # 从reconstructed_left中提取X'（前x_dim维度）
    real_x = left_input[:, :x_dim]
    reconstructed_x = reconstructed_left[:, :x_dim]
    
    # 使用NMSE损失，使量纲与其他损失项一致
    x_mse = torch.mean((real_x - reconstructed_x) ** 2)
    x_rms = torch.sqrt(torch.mean(real_x ** 2) + 1e-8)
    x_loss = x_mse / (x_rms ** 2 + 1e-8)
    
    # 3. Z分布约束：predicted_z与标准高斯分布的MMD差异
    z_target = torch.randn_like(predicted_z).to(device)  # 标准高斯分布
    z_loss = mmd_loss(predicted_z, z_target)
    
    # 总损失：组合三个损失项
    total_loss = weight_y * y_loss + weight_x * x_loss + weight_z * z_loss
    
    return {
        "total_loss": total_loss,
        "y_loss": y_loss,
        "x_loss": x_loss,
        "z_loss": z_loss
    }

# ============== 训练循环 ==============
if not skip_training:
    print('开始训练...')
    start_time = datetime.now()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        epoch_train_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
        
        optimizer.zero_grad()  # 初始清零梯度
        
        for i, batch in enumerate(train_loader):
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 计算损失
            losses = calculate_loss(left_batch, right_batch)
            
            # 梯度更新（使用梯度累积）
            scaled_loss = losses["total_loss"] / grad_accum_steps
            scaled_loss.backward()
            
            # 损失累加
            for key in epoch_train_losses:
                epoch_train_losses[key] += losses[key].item()
            
            # 每grad_accum_steps个batch进行一次梯度裁剪和参数更新
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                optimizer.step()
                optimizer.zero_grad()
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in epoch_train_losses:
            epoch_train_losses[key] /= num_batches
        
        # 记录训练历史
        train_losses['total'].append(epoch_train_losses['total_loss'])
        train_losses['y_loss'].append(epoch_train_losses['y_loss'])
        train_losses['x_loss'].append(epoch_train_losses['x_loss'])
        train_losses['z_loss'].append(epoch_train_losses['z_loss'])
        
        # 打印训练日志
        epoch_train_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_losses['total_loss']:.6f}, "
              f"Y Loss: {epoch_train_losses['y_loss']:.6f}, "
              f"X Loss: {epoch_train_losses['x_loss']:.6f}, "
              f"Z Loss: {epoch_train_losses['z_loss']:.6f}, "
              f"Time: {epoch_train_time:.2f}s")
        
        # 验证阶段
        val_start_time = time.time()
        
        model.eval()
        epoch_val_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                left_batch = batch[0].to(device)
                right_batch = batch[1].to(device)
                
                # 计算损失
                losses = calculate_loss(left_batch, right_batch)
                
                for key in epoch_val_losses:
                    epoch_val_losses[key] += losses[key].item()
        
        # 计算验证集平均损失
        for key in epoch_val_losses:
            epoch_val_losses[key] /= len(val_loader)
        
        # 计算验证时间
        val_time = time.time() - val_start_time
        
        # 打印验证日志
        print(f"          Val Loss: {epoch_val_losses['total_loss']:.6f}, "
              f"Y Loss: {epoch_val_losses['y_loss']:.6f}, "
              f"X Loss: {epoch_val_losses['x_loss']:.6f}, "
              f"Z Loss: {epoch_val_losses['z_loss']:.6f}, "
              f"Time: {val_time:.2f}s")
        
        # 记录验证历史
        val_losses['total'].append(epoch_val_losses['total_loss'])
        val_losses['y_loss'].append(epoch_val_losses['y_loss'])
        val_losses['x_loss'].append(epoch_val_losses['x_loss'])
        val_losses['z_loss'].append(epoch_val_losses['z_loss'])
        
        # 更新学习率调度器（基于验证损失）
        scheduler.step(epoch_val_losses['total_loss'])
        
        # 保存最佳模型
        if epoch_val_losses['total_loss'] < best_val_loss:
            best_val_loss = epoch_val_losses['total_loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_losses['total_loss'],
                'val_loss': epoch_val_losses['total_loss'],
                'x_mean': x_mean,
                'x_std': x_std,
                'y_mean': y_mean,
                'y_std': y_std,
                'x_dim': x_dim,
                'y_dim': y_dim,
                'z_dim': z_dim,
                'left_input_dim': left_input_dim,
                'right_input_dim': right_input_dim
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f'  -> 保存最佳模型 (Val Loss: {best_val_loss:.6f})')
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f'\n早停触发! 验证损失连续{patience}个epoch没有改善')
            break

    total_time = datetime.now() - start_time
    print(f'\n训练完成! 总时间: {total_time}')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    
    # 计算验证集上的NMSE
    print('\n=== 计算验证集NMSE ===')
    model.eval()
    total_val_nmse = 0.0
    with torch.no_grad():
        for batch in val_loader:
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 正向映射：left_input → predicted_right
            predicted_right, _, _ = model(left_batch, return_intermediate=True)
            
            # 从predicted_right中提取Y'
            predicted_y = predicted_right[:, :y_dim]
            
            # 从right_input中提取真实Y
            real_y = right_batch[:, :y_dim]
            
            # 计算NMSE
            batch_nmse = nmse_loss(real_y, predicted_y)
            total_val_nmse += batch_nmse.item()
    
    avg_val_nmse = total_val_nmse / len(val_loader)
    print(f'验证集平均NMSE: {avg_val_nmse:.6f}')
else:
    print('跳过训练阶段 (skip_training=True)')
    # 生成虚拟损失数据以避免可视化错误
    train_losses['total'] = [1.0, 0.8, 0.6, 0.4, 0.2] * 20
    train_losses['y_loss'] = [0.8, 0.6, 0.4, 0.3, 0.2] * 20
    train_losses['x_loss'] = [0.5, 0.4, 0.3, 0.2, 0.1] * 20
    train_losses['z_loss'] = [0.3, 0.2, 0.2, 0.1, 0.1] * 20
    val_losses['total'] = [1.2, 0.9, 0.7, 0.5, 0.3] * 20
    val_losses['y_loss'] = [0.9, 0.7, 0.5, 0.4, 0.3] * 20
    val_losses['x_loss'] = [0.6, 0.5, 0.4, 0.3, 0.2] * 20
    val_losses['z_loss'] = [0.4, 0.3, 0.3, 0.2, 0.2] * 20
    best_val_loss = 0.2
    avg_val_nmse = 0.2  # 虚拟值

# ============== 可视化训练曲线 ==============
print('\n=== Generating training curves ===')

fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# Total loss curve
axes[0].plot(train_losses['total'], label='Train Total', color='blue')
axes[0].plot(val_losses['total'], label='Val Total', color='red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Y loss curve (NMSE)
axes[1].plot(train_losses['y_loss'], label='Train Y Loss', color='blue')
axes[1].plot(val_losses['y_loss'], label='Val Y Loss', color='red')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Y Loss (NMSE)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# X loss curve (MSE)
axes[2].plot(train_losses['x_loss'], label='Train X Loss', color='blue')
axes[2].plot(val_losses['x_loss'], label='Val X Loss', color='red')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('X Loss (MSE)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Z loss curve (MMD)
axes[3].plot(train_losses['z_loss'], label='Train Z Loss', color='blue')
axes[3].plot(val_losses['z_loss'], label='Val Z Loss', color='red')
axes[3].set_xlabel('Epoch')
axes[3].set_ylabel('Loss')
axes[3].set_title('Z Loss (MMD)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_losses.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Training curves saved to:', os.path.join(checkpoint_dir, 'training_losses.png'))

# ============== 模型功能实现：固定x预测y ==============
print('\n=== Fixed x predicting y functionality ===')

# 从验证集中选取五个测试样本
test_indices = [0, 10, 20, 30, 40]  # 五个不同的测试样本索引

for i, test_idx in enumerate(test_indices):
    print(f'\nPredicting y for test sample {i+1}:')
    
    x_test = val_x_normalized[test_idx:test_idx+1]  # 形状：(1, x_dim)

    # 左侧输入：X + 零填充
    left_test_input = np.concatenate((x_test, np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
    left_test_input = torch.FloatTensor(left_test_input).to(device)

    # 使用模型进行正向预测
    model.eval()
    with torch.no_grad():
        predicted_right, _, _ = model(left_test_input, return_intermediate=True)
        
        # 从predicted_right中提取Y'
        predicted_y_normalized = predicted_right[:, :y_dim]
        
        # 反标准化得到预测的y
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean

    # 获取真实的y值
    real_y = val_y[test_idx:test_idx+1]

    # 计算NMSE
    real_y_tensor = torch.FloatTensor(real_y).to(device)
    predicted_y_tensor = torch.FloatTensor(predicted_y).to(device)
    # 计算NMSE时需要归一化数据
    real_y_normalized = (real_y - y_mean) / (y_std + 1e-8)
    predicted_y_normalized = (predicted_y - y_mean) / (y_std + 1e-8)
    real_y_normalized_tensor = torch.FloatTensor(real_y_normalized).to(device)
    predicted_y_normalized_tensor = torch.FloatTensor(predicted_y_normalized).to(device)
    nmse_value = nmse_loss(predicted_y_normalized_tensor, real_y_normalized_tensor).item()

    print(f'  Test sample {i+1} prediction result:')
    print(f'    Predicted y shape: {predicted_y.shape}')
    print(f'    NMSE: {nmse_value:.6f}')

    # 可视化预测结果 - 确保频率点和y数据点数量一致
    plt.figure(figsize=(10, 6))
    plt.plot(freq_data[:y_dim], real_y[0], label='Real y', color='blue', linewidth=2)
    plt.plot(freq_data[:y_dim], predicted_y[0], label='Predicted y', color='red', linestyle='--', linewidth=2)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    plt.title(f'Comparison of y prediction with fixed x - Sample {i+1}')
    # 添加NMSE值到图表
    plt.text(0.02, 0.02, f'NMSE: {nmse_value:.6f}', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'fixed_x_predicted_y_{i+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved for sample {i+1}: fixed_x_predicted_y_{i+1}.png')

# ============== 模型功能实现：固定y回推x ==============
print('\n=== Fixed y backward predicting x functionality ===')

# 从验证集中选取五个测试样本
y_test_indices = [0, 10, 20, 30, 40]  # 五个不同的测试样本索引

for i, y_test_idx in enumerate(y_test_indices):
    print(f'\nBackward predicting x for test sample {i+1}:')
    
    # 从验证集中选取一个测试样本
    y_test = val_y_normalized[y_test_idx:y_test_idx+1]  # 形状：(1, y_dim)

    # 创建右侧输入：Y + Z（Z是随机生成的标准高斯分布）
    z_test = np.random.randn(1, z_dim).astype(np.float32)
    right_test_input = np.concatenate((y_test, z_test), axis=1)
    right_test_input = torch.FloatTensor(right_test_input).to(device)

    # 使用模型进行反向预测
    with torch.no_grad():
        reconstructed_left, _ = model.inverse(right_test_input)
        
        # 从reconstructed_left中提取X'
        reconstructed_x_normalized = reconstructed_left[:, :x_dim]
        
        # 反标准化得到回推的x
        reconstructed_x = reconstructed_x_normalized.cpu().numpy() * x_std + x_mean

    # 获取真实的x值
    real_x = val_x[y_test_idx:y_test_idx+1]

    print(f'  Test sample {i+1} backward result:')
    print(f'    Real x: {real_x[0]}')
    print(f'    Backward x: {reconstructed_x[0]}')

    # 使用回推的x进行正向预测，验证一致性
    with torch.no_grad():
        # 左侧输入：回推的X + 零填充
        left_test_input = np.concatenate((reconstructed_x_normalized.cpu().numpy(), np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
        left_test_input = torch.FloatTensor(left_test_input).to(device)
        
        # 正向预测
        predicted_right, _, _ = model(left_test_input, return_intermediate=True)
        predicted_y_normalized = predicted_right[:, :y_dim]
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean

    # 获取真实的y值
    real_y = val_y[y_test_idx:y_test_idx+1]

    # 可视化：将x分布和y预测拼到一起
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # 第一个子图：x参数对比
    grid = np.arange(x_dim)
    width = 0.35
    
    ax1.bar(grid - width/2, real_x[0], width, label='Real x', color='blue')
    ax1.bar(grid + width/2, reconstructed_x[0], width, label='Backward x', color='red')
    ax1.set_xlabel('Geometry Parameter Index')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title(f'X Parameters Comparison - Sample {i+1}')
    ax1.set_xticks(grid)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：y预测对比
    ax2.plot(freq_data[:y_dim], real_y[0], label='Real y', color='blue', linewidth=2)
    ax2.plot(freq_data[:y_dim], predicted_y[0], label='Predicted y (from backward x)', color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S11 (dB)')
    ax2.set_title(f'Y Prediction Consistency - Sample {i+1}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'fixed_y_backward_x_{i+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved for sample {i+1}: fixed_y_backward_x_{i+1}.png')

# ============== 多解生成功能测试 ==============
print('\n=== Multiple solutions generation functionality test ===')

# 选择一个特定的测试样本进行多解生成
multi_solution_idx = 0
y_test = val_y_normalized[multi_solution_idx:multi_solution_idx+1]  # 形状：(1, y_dim)
real_x = val_x[multi_solution_idx:multi_solution_idx+1]  # 真实的x值

# 对于同一个y，生成多个z样本
num_samples = 100  # 100 prediction samples
z_scale = 1.2  # Adjust Z sampling range to increase diversity and improve result fit
z_samples = np.random.randn(num_samples, z_dim).astype(np.float32) * z_scale

# 为每个z样本创建右侧输入
y_test_repeated = np.repeat(y_test, num_samples, axis=0)  # Repeat y to match z_samples quantity
right_test_inputs = np.concatenate((y_test_repeated, z_samples), axis=1)
right_test_inputs = torch.FloatTensor(right_test_inputs).to(device)

# 使用模型进行批量反向预测
with torch.no_grad():
    reconstructed_lefts, _ = model.inverse(right_test_inputs)
    
    # 从reconstructed_lefts中提取X'
    reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
    
    # 反标准化得到回推的x样本
    reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_std + x_mean

# 对生成的X进行后处理，确保在合理物理范围内
x_min = x_features.min(axis=0)
x_max = x_features.max(axis=0)
reconstructed_xs_clipped = np.clip(reconstructed_xs, x_min, x_max)

# 验证X的多样性
print(f'\nMultiple solutions generation results:')
print(f'  Number of generated solutions: {num_samples}')
print(f'  Real x: {real_x[0]}')

# 计算生成的X之间的多样性（标准差）
diversity = np.std(reconstructed_xs_clipped, axis=0)
print(f'\nX diversity (standard deviation for each parameter): {diversity}')
print(f'  Average diversity: {np.mean(diversity)}')

# 验证生成的X的正确性：使用生成的X进行正向预测，检查是否接近原始Y
print('\nVerifying correctness of generated X:')

# 对每个生成的X进行正向预测
generated_xs_normalized = (reconstructed_xs_clipped - x_mean) / (x_std + 1e-8)
left_predict_inputs = np.concatenate((generated_xs_normalized, np.zeros((num_samples, padding_dim), dtype=np.float32)), axis=1)
left_predict_inputs = torch.FloatTensor(left_predict_inputs).to(device)

with torch.no_grad():
    predicted_rights, _ = model(left_predict_inputs)
    
    # 从predicted_rights中提取Y'
    predicted_ys_normalized = predicted_rights[:, :y_dim]
    
    # 反标准化得到预测的y
    predicted_ys = predicted_ys_normalized.cpu().numpy() * y_std + y_mean

# 计算每个预测的Y与原始Y的误差
y_test_original = val_y[multi_solution_idx:multi_solution_idx+1]  # Original unnormalized Y
errors = []
for i in range(num_samples):
    error = np.mean(np.abs(predicted_ys[i] - y_test_original[0]))
    errors.append(error)
    # 只打印前10个解的误差，避免输出过多
    if i < 10:
        print(f'  Solution {i+1} Y prediction error: {error:.6f}')
if num_samples > 10:
    print(f'  ... and {num_samples - 10} more solutions')

# 排序误差，获取前5个最小误差的索引
errors = np.array(errors)
top_indices = np.argsort(errors)[:5]
print(f'\nTop 5 solutions with smallest error indices: {top_indices + 1}')
print(f'Corresponding error values: {errors[top_indices]}')

# 可视化生成的X的分布
plt.figure(figsize=(12, 8))

for param_idx in range(x_dim):
    plt.subplot(2, 3, param_idx + 1)
    plt.hist(reconstructed_xs_clipped[:, param_idx], bins=10, alpha=0.7, color='green', label='Generated X')
    plt.axvline(real_x[0, param_idx], color='red', linestyle='--', label='Real X')
    plt.title(f'Parameter {param_idx + 1} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'multi_solution_x_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# 可视化生成的X对应的Y预测（只显示误差最小的5个）
plt.figure(figsize=(12, 6))
plt.plot(freq_data, y_test_original[0], label='Original Y', color='blue', linewidth=2)

for i, idx in enumerate(top_indices):
    plt.plot(freq_data, predicted_ys[idx], label=f'Predicted Y (Top {i+1}, Error: {errors[idx]:.6f})', alpha=0.7)

plt.xlabel('Frequency (GHz)')
plt.ylabel('S11')
plt.title('Original Y vs Top 5 Predicted Y from Generated X')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'multi_solution_y_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()

# 保存多解生成结果
np.save(os.path.join(checkpoint_dir, 'generated_xs.npy'), reconstructed_xs)
np.save(os.path.join(checkpoint_dir, 'predicted_ys.npy'), predicted_ys)

# ============== Saving prediction results ==============
# Note: Prediction results for fixed x predicting y and fixed y backward predicting x have already been saved inside their respective loops

print('\n=== Training completed! ===')
print(f'Model checkpoints saved in: {checkpoint_dir}')
print('Prediction results have been saved.')