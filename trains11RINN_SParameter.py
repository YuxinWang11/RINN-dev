"""
RINN Model Training Script - S-parameter Dataset
Task: Use 6-dimensional input (5 geometric parameters + frequency) to predict 2-dimensional output (real part RE + imaginary part IM)
Left input: 6-dimensional (H1, H2, H3, H_C1, H_C2, frequency)
Right input: 2-dimensional (RE, IM) + 4-dimensional zero padding = 6-dimensional
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
    parser = argparse.ArgumentParser(description='RINN Model Training Script - S-parameter Dataset')
    parser.add_argument('--config', type=str, default=None, help='Configuration file path')
    return parser.parse_args()

args = parse_args()

# 加载配置
config = {
    "model_config": {
        "hidden_dim": 32,
        "num_blocks": 3,
        "num_stages": 2,
        "num_cycles_per_stage": 2,
        "ratio_toZ_after_flowstage": 0.3,
        "ratio_x1_x2_inAffine": None  # 会在后面计算
    },
    "training_params": {
        "batch_size": 128,
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
        "normalization_method": "standard",
        "geometry_sample_ratio": 0.1  # 几何参数组合采样比例（默认10%）
    }
}

# If config file is provided, load configuration
if args.config and os.path.exists(args.config):
    print(f'Loading parameters from config file: {args.config}')
    with open(args.config, 'r', encoding='utf-8') as f:
        loaded_config = json.load(f)
    # Update configuration
    if 'model_config' in loaded_config:
        config['model_config'].update(loaded_config['model_config'])
    if 'training_params' in loaded_config:
        config['training_params'].update(loaded_config['training_params'])
    if 'data_params' in loaded_config:
        config['data_params'].update(loaded_config['data_params'])
    print('Configuration loaded successfully!')

# ============== Create training output folder ==============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"rinn_sparameter_{timestamp}"
checkpoint_dir = os.path.join('model_checkpoints_rinn', experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f'Training output folder: {checkpoint_dir}')

# 保存使用的配置
with open(os.path.join(checkpoint_dir, 'used_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设备配置
from R_INN_model.device_utils import get_device
device = get_device()

# ============== Data loading and preprocessing ==============
print('\n=== Loading data ===')

# Load S-parameter CSV file
data_path = 'data/S Parameter Plot 7.csv'

# Read CSV file
with open(data_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([row for row in reader], dtype=np.float32)

# Extract frequency data
freq_data = data[:, 0]
print(f'Number of frequency points: {len(freq_data)}')
print(f'Frequency range: {freq_data[0]} GHz - {freq_data[-1]} GHz')

# Extract geometry parameters and data
def extract_geometry_params(col_name):
    """Extract geometry parameters H1, H2, H3, H_C1, H_C2 from column name"""
    col_name = col_name.replace('\n', '').replace('"', '')
    
    h1_match = re.search(r"H1='([\d.]+)mm'", col_name)
    h2_match = re.search(r"H2='([\d.]+)mm'", col_name)
    h3_match = re.search(r"H3='([\d.]+)mm'", col_name)
    hc1_match = re.search(r"H_C1='([\d.]+)mm'", col_name)
    hc2_match = re.search(r"H_C2='([\d.]+)mm'", col_name)
    
    if all([h1_match, h2_match, h3_match, hc1_match, hc2_match]):
        return {
            'H1': float(h1_match.group(1)),
            'H2': float(h2_match.group(1)),
            'H3': float(h3_match.group(1)),
            'H_C1': float(hc1_match.group(1)),
            'H_C2': float(hc2_match.group(1))
        }
    return None

# Store all samples
samples = []

# Process each column data
header_cols = header[1:]
for col_idx, col_name in enumerate(header_cols):
    # Extract geometry parameters
    params = extract_geometry_params(col_name)
    if params is None:
        continue
    
    # Determine if it's real or imaginary part
    is_real = 're(S(1,1))' in col_name
    is_imag = 'im(S(1,1))' in col_name
    
    # Extract data from this column
    col_data = data[:, col_idx + 1]
    
    # Create a sample for each frequency point
    for i, freq in enumerate(freq_data):
        sample = {
            'H1': params['H1'],
            'H2': params['H2'],
            'H3': params['H3'],
            'H_C1': params['H_C1'],
            'H_C2': params['H_C2'],
            'freq': freq,
            'value': col_data[i],
            'type': 'real' if is_real else 'imaginary'
        }
        samples.append(sample)

print(f'Original number of samples: {len(samples)}')

# Group by geometry parameters and frequency, pair real and imaginary parts
paired_samples = {}
for sample in samples:
    key = (sample['H1'], sample['H2'], sample['H3'], 
           sample['H_C1'], sample['H_C2'], sample['freq'])
    
    if key not in paired_samples:
        paired_samples[key] = {'real': None, 'imag': None}
    
    if sample['type'] == 'real':
        paired_samples[key]['real'] = sample['value']
    else:
        paired_samples[key]['imag'] = sample['value']

# Filter out samples with only real or only imaginary part
valid_samples = []
for key, values in paired_samples.items():
    if values['real'] is not None and values['imag'] is not None:
        valid_samples.append({
            'H1': key[0],
            'H2': key[1],
            'H3': key[2],
            'H_C1': key[3],
            'H_C2': key[4],
            'freq': key[5],
            'real': values['real'],
            'imag': values['imag']
        })

print(f'Number of paired samples: {len(valid_samples)}')

# ============== New: Use only one-tenth of geometry parameter combinations ==============
print('\n=== Geometry parameter combination sampling ===')

# Extract all unique geometry parameter combinations
geometry_combinations = {}
for sample in valid_samples:
    key = (sample['H1'], sample['H2'], sample['H3'], sample['H_C1'], sample['H_C2'])
    if key not in geometry_combinations:
        geometry_combinations[key] = []
    geometry_combinations[key].append(sample)

print(f'Total number of geometry parameter combinations: {len(geometry_combinations)}')

# Get sampling ratio from configuration
sample_ratio = config['data_params'].get('geometry_sample_ratio', 0.1)

# Randomly select specified ratio of geometry parameter combinations
num_combinations = len(geometry_combinations)
num_selected = max(1, int(num_combinations * sample_ratio))  # Select at least 1 combination
all_keys = list(geometry_combinations.keys())
# Use index selection because np.random.choice can't directly handle tuple lists
selected_indices = np.random.choice(len(all_keys), num_selected, replace=False)
selected_keys = [all_keys[i] for i in selected_indices]

print(f'Selected number of geometry parameter combinations: {num_selected} (占总数的 {num_selected/num_combinations*100:.1f}%)')

# Extract all samples from selected geometry parameter combinations
selected_samples = []
for key in selected_keys:
    selected_samples.extend(geometry_combinations[key])

print(f'Number of samples after sampling: {len(selected_samples)}')

valid_samples = selected_samples

# 转换为数组格式
X_data = []
Y_data = []

for sample in valid_samples:
    # 输入：6维（H1, H2, H3, H_C1, H_C2, 频率）
    x = [
        sample['H1'],
        sample['H2'],
        sample['H3'],
        sample['H_C1'],
        sample['H_C2'],
        sample['freq']
    ]
    
    # 输出：2维（实部, 虚部）
    y = [
        sample['real'],
        sample['imag']
    ]
    
    X_data.append(x)
    Y_data.append(y)

X_data = np.array(X_data, dtype=np.float32)
Y_data = np.array(Y_data, dtype=np.float32)

print(f'X数据形状: {X_data.shape}')
print(f'Y数据形状: {Y_data.shape}')
print(f'X特征示例 (第一个样本): {X_data[0]}')
print(f'Y特征示例 (第一个样本): {Y_data[0]}')

# 数据标准化：使用标准归一化（均值-标准差）
normalization_method = config['data_params']['normalization_method']  # 'standard'

# 先划分训练集和验证集的索引，再分别进行归一化
n_samples = len(X_data)
indices = np.random.permutation(n_samples)
train_size = int(0.8 * n_samples)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 提取训练集数据
train_x = X_data[train_indices]
train_y = Y_data[train_indices]

# 提取验证集数据
val_x = X_data[val_indices]
val_y = Y_data[val_indices]

# 使用均值-标准差归一化
x_mean = train_x.mean(axis=0)
x_std = train_x.std(axis=0)
train_x_normalized = (train_x - x_mean) / (x_std + 1e-8)
val_x_normalized = (val_x - x_mean) / (x_std + 1e-8)

y_mean = train_y.mean(axis=0)
y_std = train_y.std(axis=0)
train_y_normalized = (train_y - y_mean) / (y_std + 1e-8)
val_y_normalized = (val_y - y_mean) / (y_std + 1e-8)

# 合并归一化后的训练集和验证集
X_features_normalized = np.zeros_like(X_data, dtype=np.float32)
X_features_normalized[train_indices] = train_x_normalized
X_features_normalized[val_indices] = val_x_normalized

Y_features_normalized = np.zeros_like(Y_data, dtype=np.float32)
Y_features_normalized[train_indices] = train_y_normalized
Y_features_normalized[val_indices] = val_y_normalized

# 数据质量检查
print(f'归一化方法: {normalization_method}')
print(f'X特征归一化后均值: {X_features_normalized.mean(axis=0).mean():.6f}, 标准差: {X_features_normalized.std(axis=0).mean():.6f}')
print(f'Y数据归一化后均值: {Y_features_normalized.mean(axis=0).mean():.6f}, 标准差: {Y_features_normalized.std(axis=0).mean():.6f}')

# 检查是否存在NaN或无穷大值
print(f'X特征是否包含NaN: {np.isnan(X_features_normalized).any()}')
print(f'X特征是否包含无穷大: {np.isinf(X_features_normalized).any()}')
print(f'Y数据是否包含NaN: {np.isnan(Y_features_normalized).any()}')
print(f'Y数据是否包含无穷大: {np.isinf(Y_features_normalized).any()}')

# 检查归一化后的数据范围
print(f'X特征归一化后最小值: {X_features_normalized.min():.6f}, 最大值: {X_features_normalized.max():.6f}')
print(f'Y数据归一化后最小值: {Y_features_normalized.min():.6f}, 最大值: {Y_features_normalized.max():.6f}')

# ============== 维度处理：正确的数据结构 ==============
print('\n=== 维度处理与数据结构 ===')

# 配置参数
x_dim = X_data.shape[1]  # X维度：6
y_dim = Y_data.shape[1]       # Y维度：2
z_dim = x_dim                 # Z维度：6（与X维度相同）

# 左侧输入：X + 零填充 → 总维度 = x_dim + padding_dim = x_dim + y_dim = 8
padding_dim = y_dim
left_input_dim = x_dim + padding_dim

# 右侧输入：Y + Z → 总维度 = y_dim + z_dim = 8
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
        'hidden_dim': config['model_config']['hidden_dim'],
        'num_blocks': config['model_config']['num_blocks'],
        'num_stages': config['model_config']['num_stages'],
        'num_cycles_per_stage': config['model_config']['num_cycles_per_stage'],
        'ratio_toZ_after_flowstage': ratio_toZ_after_flowstage,
        'ratio_x1_x2_inAffine': ratio_x1_x2_inAffine
    },
    'training_params': {
        'batch_size': config['training_params']['batch_size'],
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

with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
    json.dump(training_info, f, ensure_ascii=False, indent=2)

# ============== 损失计算函数 ==============
def calculate_nmse(predictions, targets):
    """计算归一化均方误差 (NMSE)"""
    mse = torch.mean((predictions - targets) ** 2)
    signal_power = torch.mean(targets ** 2) + 1e-8
    nmse = mse / signal_power
    return nmse

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
        epoch_train_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0, "y_nmse": 0.0}
        
        optimizer.zero_grad()  # 初始清零梯度
        
        # 创建训练数据集和加载器
        train_dataset = TensorDataset(left_train, right_train)
        train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True)
        
        for i, batch in enumerate(train_loader):
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 计算损失
            losses = calculate_loss(left_batch, right_batch)
            
            # 计算NMSE指标
            predicted_right, _, _ = model(left_batch, return_intermediate=True)
            predicted_y = predicted_right[:, :y_dim]
            real_y = right_batch[:, :y_dim]
            y_nmse = calculate_nmse(predicted_y, real_y)
            
            # 梯度更新（使用梯度累积）
            scaled_loss = losses["total_loss"] / grad_accum_steps
            scaled_loss.backward()
            
            # 损失累加（只累加losses字典中存在的键）
            for key in losses:
                epoch_train_losses[key] += losses[key].item()
            
            # NMSE累加
            epoch_train_losses['y_nmse'] += y_nmse.item()
            
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
        train_losses['y_nmse'] = train_losses.get('y_nmse', [])
        train_losses['y_nmse'].append(epoch_train_losses.get('y_nmse', 0.0))
        
        # 打印训练日志
        epoch_train_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_losses['total_loss']:.6f}, "
              f"Y Loss: {epoch_train_losses['y_loss']:.6f}, "
              f"Y NMSE: {epoch_train_losses.get('y_nmse', 0.0):.6f}, "
              f"X Loss: {epoch_train_losses['x_loss']:.6f}, "
              f"Z Loss: {epoch_train_losses['z_loss']:.6f}, "
              f"Time: {epoch_train_time:.2f}s")
        
        # 验证阶段
        val_start_time = time.time()
        
        model.eval()
        epoch_val_losses = {"total_loss": 0.0, "y_loss": 0.0, "x_loss": 0.0, "z_loss": 0.0, "y_nmse": 0.0}
        
        # 创建验证数据集和加载器
        val_dataset = TensorDataset(left_val, right_val)
        val_loader = DataLoader(val_dataset, batch_size=config['training_params']['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in val_loader:
                left_batch = batch[0].to(device)
                right_batch = batch[1].to(device)
                
                # 计算损失
                losses = calculate_loss(left_batch, right_batch)
                
                # 计算NMSE指标
                predicted_right, _, _ = model(left_batch, return_intermediate=True)
                predicted_y = predicted_right[:, :y_dim]
                real_y = right_batch[:, :y_dim]
                y_nmse = calculate_nmse(predicted_y, real_y)
                
                # 损失累加（只累加losses字典中存在的键）
                for key in losses:
                    epoch_val_losses[key] += losses[key].item()
                
                # NMSE累加
                epoch_val_losses['y_nmse'] += y_nmse.item()
        
        # 计算验证集平均损失
        num_val_batches = len(val_loader)
        for key in epoch_val_losses:
            epoch_val_losses[key] /= num_val_batches
        
        # 记录验证历史
        val_losses['total'].append(epoch_val_losses['total_loss'])
        val_losses['y_loss'].append(epoch_val_losses['y_loss'])
        val_losses['x_loss'].append(epoch_val_losses['x_loss'])
        val_losses['z_loss'].append(epoch_val_losses['z_loss'])
        val_losses['y_nmse'] = val_losses.get('y_nmse', [])
        val_losses['y_nmse'].append(epoch_val_losses.get('y_nmse', 0.0))
        
        # 打印验证日志
        epoch_val_time = time.time() - val_start_time
        print(f"          Val Loss: {epoch_val_losses['total_loss']:.6f}, "
              f"Y Loss: {epoch_val_losses['y_loss']:.6f}, "
              f"Y NMSE: {epoch_val_losses.get('y_nmse', 0.0):.6f}, "
              f"X Loss: {epoch_val_losses['x_loss']:.6f}, "
              f"Z Loss: {epoch_val_losses['z_loss']:.6f}, "
              f"Time: {epoch_val_time:.2f}s")
        
        # 学习率调度
        scheduler.step(epoch_val_losses['total_loss'])
        
        # 早停检查
        if epoch_val_losses['total_loss'] < best_val_loss:
            best_val_loss = epoch_val_losses['total_loss']
            patience_counter = 0
            
            # 保存最佳模型
            model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_losses['total_loss'],
                'val_loss': epoch_val_losses['total_loss'],
                'config': config,
                'stats': {
                    'x_mean': x_mean.tolist(),
                    'x_std': x_std.tolist(),
                    'y_mean': y_mean.tolist(),
                    'y_std': y_std.tolist()
                }
            }, model_path)
            print(f"  -> 最佳模型已保存 (Val Loss: {epoch_val_losses['total_loss']:.6f})")
        else:
            patience_counter += 1
        
        # 早停触发
        if patience_counter >= patience:
            print(f"早停触发，停止训练 (patience: {patience_counter})")
            break
    
    # 训练完成
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"\n训练完成! 总训练时间: {training_duration}")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 绘制训练曲线
    plt.figure(figsize=(18, 12))
    
    # 总损失
    plt.subplot(2, 3, 1)
    plt.plot(train_losses['total'], label='Train Loss')
    plt.plot(val_losses['total'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Y损失
    plt.subplot(2, 3, 2)
    plt.plot(train_losses['y_loss'], label='Train Y Loss')
    plt.plot(val_losses['y_loss'], label='Val Y Loss')
    plt.title('Y Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Y NMSE
    plt.subplot(2, 3, 3)
    plt.plot(train_losses['y_nmse'], label='Train Y NMSE')
    plt.plot(val_losses['y_nmse'], label='Val Y NMSE')
    plt.title('Y NMSE (Normalized Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.legend()
    plt.grid(True)
    
    # X损失
    plt.subplot(2, 3, 4)
    plt.plot(train_losses['x_loss'], label='Train X Loss')
    plt.plot(val_losses['x_loss'], label='Val X Loss')
    plt.title('X Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Z损失
    plt.subplot(2, 3, 5)
    plt.plot(train_losses['z_loss'], label='Train Z Loss')
    plt.plot(val_losses['z_loss'], label='Val Z Loss')
    plt.title('Z Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 所有损失对比
    plt.subplot(2, 3, 6)
    plt.plot(train_losses['total'], label='Train Total', alpha=0.7)
    plt.plot(val_losses['total'], label='Val Total', alpha=0.7)
    plt.plot(train_losses['y_nmse'], label='Train Y NMSE', alpha=0.7, linestyle='--')
    plt.plot(val_losses['y_nmse'], label='Val Y NMSE', alpha=0.7, linestyle='--')
    plt.title('All Losses Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/NMSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(checkpoint_dir, 'training_losses.png')
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {loss_plot_path}")
    plt.close()

# ============== 多解生成测试 ==============
print('\n=== 多解生成测试 ===')

# 加载最佳模型
model_path = os.path.join(checkpoint_dir, 'best_model.pth')
if os.path.exists(model_path):
    print(f"加载最佳模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载完成，训练到第 {checkpoint['epoch']+1} 轮")
else:
    print("未找到最佳模型，使用当前模型进行测试")

model.eval()

# 测试前向预测：给定X，预测Y
print('\n测试前向预测（给定X，预测Y）:')

# 选择几个测试样本
num_test_samples = 5
test_indices = np.random.choice(len(left_val), num_test_samples, replace=False)
test_left_input = left_val[test_indices].to(device)
test_real_x = test_left_input[:, :x_dim]

with torch.no_grad():
    # 前向预测
    predicted_right, _, _ = model(test_left_input, return_intermediate=True)
    predicted_y = predicted_right[:, :y_dim]
    
    # 反标准化
    predicted_y_denorm = predicted_y.cpu().numpy() * y_std + y_mean
    real_x_denorm = test_real_x.cpu().numpy() * x_std + x_mean

print(f"前向预测结果（展示 {num_test_samples} 个样本）:")
for i in range(num_test_samples):
    print(f"样本 {i+1}:")
    print(f"  输入X: {real_x_denorm[i]}")
    print(f"  预测Y: {predicted_y_denorm[i]}")

# 测试反向生成：给定Y，生成多个可能的X
print('\n测试反向生成（给定Y，生成多个可能的X）:')

# 选择一个测试样本的Y值
num_gen_samples = 100
test_y_idx = np.random.choice(len(right_val), 1, replace=False)[0]
test_real_y = right_val[test_y_idx, :y_dim].to(device)

# 生成多个Z值
z_samples = torch.randn(num_gen_samples, z_dim).to(device)

# 构建右侧输入：Y + Z
right_inputs = torch.cat([test_real_y.repeat(num_gen_samples, 1), z_samples], dim=1)

with torch.no_grad():
    # 反向映射
    generated_left, _ = model.inverse(right_inputs)
    generated_x = generated_left[:, :x_dim]
    
    # 反标准化
    generated_x_denorm = generated_x.cpu().numpy() * x_std + x_mean
    real_y_denorm = test_real_y.cpu().numpy() * y_std + y_mean

print(f"反向生成结果（给定Y: {real_y_denorm}，生成 {num_gen_samples} 个可能的X）:")
print(f"生成的X样本统计:")
print(f"  均值: {generated_x_denorm.mean(axis=0)}")
print(f"  标准差: {generated_x_denorm.std(axis=0)}")
print(f"  最小值: {generated_x_denorm.min(axis=0)}")
print(f"  最大值: {generated_x_denorm.max(axis=0)}")

# 保存生成的样本
np.save(os.path.join(checkpoint_dir, 'generated_xs.npy'), generated_x_denorm)
np.save(os.path.join(checkpoint_dir, 'predicted_ys.npy'), predicted_y_denorm)
print(f"\n生成的样本已保存到:")
print(f"  {os.path.join(checkpoint_dir, 'generated_xs.npy')}")
print(f"  {os.path.join(checkpoint_dir, 'predicted_ys.npy')}")

print('\n=== 训练和测试完成 ===')

# ============== 新增：单个几何参数五元组的趋势图可视化 ==============
print('\n=== 生成单个几何参数五元组的趋势图 ===')

# 选择一个几何参数五元组进行可视化
# 从验证集中选择一个几何参数组合
sample_idx = 0  # 选择验证集的第一个样本
sample_x = val_x[sample_idx]  # 原始X数据（未归一化）

# 提取几何参数五元组
geometry_params = {
    'H1': sample_x[0],
    'H2': sample_x[1],
    'H3': sample_x[2],
    'H_C1': sample_x[3],
    'H_C2': sample_x[4]
}

print(f"选择的几何参数五元组: H1={geometry_params['H1']:.2f}mm, H2={geometry_params['H2']:.2f}mm, "
      f"H3={geometry_params['H3']:.2f}mm, H_C1={geometry_params['H_C1']:.2f}mm, H_C2={geometry_params['H_C2']:.2f}mm")

# 在完整数据集中找到该几何参数五元组下所有频率点的数据
geometry_key = (geometry_params['H1'], geometry_params['H2'], geometry_params['H3'],
                geometry_params['H_C1'], geometry_params['H_C2'])

# 收集该几何参数下的所有样本
matched_indices = []
for i, sample in enumerate(valid_samples):
    if (abs(sample['H1'] - geometry_params['H1']) < 1e-6 and
        abs(sample['H2'] - geometry_params['H2']) < 1e-6 and
        abs(sample['H3'] - geometry_params['H3']) < 1e-6 and
        abs(sample['H_C1'] - geometry_params['H_C1']) < 1e-6 and
        abs(sample['H_C2'] - geometry_params['H_C2']) < 1e-6):
        matched_indices.append(i)

print(f"找到 {len(matched_indices)} 个匹配的样本")

# 提取该几何参数下的所有数据
freqs = []
real_values = []
imag_values = []

for idx in matched_indices:
    sample = valid_samples[idx]
    freqs.append(sample['freq'])
    real_values.append(sample['real'])
    imag_values.append(sample['imag'])

# 按频率排序
sort_indices = np.argsort(freqs)
freqs = np.array(freqs)[sort_indices]
real_values = np.array(real_values)[sort_indices]
imag_values = np.array(imag_values)[sort_indices]

print(f"频率范围: {freqs[0]:.2f} GHz - {freqs[-1]:.2f} GHz, 频率点数: {len(freqs)}")

# 准备模型预测的输入数据
# 构建输入：每个频率点对应一个样本，几何参数相同，频率不同
test_x_list = []
for freq in freqs:
    x = [
        geometry_params['H1'],
        geometry_params['H2'],
        geometry_params['H3'],
        geometry_params['H_C1'],
        geometry_params['H_C2'],
        freq
    ]
    test_x_list.append(x)

test_x_array = np.array(test_x_list, dtype=np.float32)

# 归一化
test_x_normalized = (test_x_array - x_mean) / (x_std + 1e-8)

# 添加零填充
test_left_input = np.concatenate((test_x_normalized, np.zeros((len(test_x_normalized), padding_dim), dtype=np.float32)), axis=1)

# 转换为torch张量
test_left_tensor = torch.FloatTensor(test_left_input).to(device)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    predicted_right, _, _ = model(test_left_tensor, return_intermediate=True)
    predicted_y = predicted_right[:, :y_dim]
    
    # 反标准化
    predicted_y_denorm = predicted_y.cpu().numpy() * y_std + y_mean

predicted_real = predicted_y_denorm[:, 0]
predicted_imag = predicted_y_denorm[:, 1]

# 绘制趋势图
plt.figure(figsize=(14, 8))

# 绘制真实值和预测值
plt.plot(freqs, real_values, 'b-', linewidth=2, label='Ground Truth S11 Real', alpha=0.8)
plt.plot(freqs, imag_values, 'r-', linewidth=2, label='Ground Truth S11 Imaginary', alpha=0.8)
plt.plot(freqs, predicted_real, 'b--', linewidth=2, label='Predicted S11 Real', alpha=0.8)
plt.plot(freqs, predicted_imag, 'r--', linewidth=2, label='Predicted S11 Imaginary', alpha=0.8)

# 设置图表标题和标签
title_str = (f"Geometry Parameters: H1={geometry_params['H1']:.2f}mm, H2={geometry_params['H2']:.2f}mm, "
             f"H3={geometry_params['H3']:.2f}mm, H_C1={geometry_params['H_C1']:.2f}mm, H_C2={geometry_params['H_C2']:.2f}mm")
plt.title(title_str, fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
plt.ylabel('S11 Value', fontsize=12, fontweight='bold')

# 设置图例
plt.legend(loc='best', fontsize=10, framealpha=0.9)

# 添加网格
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 调整布局
plt.tight_layout()

# 保存图表
trend_plot_path = os.path.join(checkpoint_dir, 's11_trend_plot.png')
plt.savefig(trend_plot_path, dpi=150, bbox_inches='tight')
print(f"Trend plot saved to: {trend_plot_path}")
plt.close()

# 计算预测误差
real_mae = np.mean(np.abs(real_values - predicted_real))
imag_mae = np.mean(np.abs(imag_values - predicted_imag))
real_rmse = np.sqrt(np.mean((real_values - predicted_real) ** 2))
imag_rmse = np.sqrt(np.mean((imag_values - predicted_imag) ** 2))

# 计算NMSE
real_nmse = np.mean((real_values - predicted_real) ** 2) / (np.mean(real_values ** 2) + 1e-8)
imag_nmse = np.mean((imag_values - predicted_imag) ** 2) / (np.mean(imag_values ** 2) + 1e-8)
overall_nmse = (real_nmse + imag_nmse) / 2

print(f"\nPrediction Error Statistics:")
print(f"  Real Part MAE: {real_mae:.6f}, RMSE: {real_rmse:.6f}, NMSE: {real_nmse:.6f}")
print(f"  Imaginary Part MAE: {imag_mae:.6f}, RMSE: {imag_rmse:.6f}, NMSE: {imag_nmse:.6f}")
print(f"  Overall NMSE: {overall_nmse:.6f}")

print('\n=== 所有任务完成 ===')