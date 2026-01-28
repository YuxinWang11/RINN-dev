"""
基于RINN模型的6维输入(5维几何参数+频率)预测1维输出(S11值)实现
正确的数据结构和损失计算：
1. 左侧输入：X(5维几何参数) + 频率(1维) + 零填充 → 总维度 = 6 + padding_dim = 6 + 1 = 7
2. 右侧输入：Y(1维S11值) + Z(6维随机噪声) → 总维度 = 1 + 6 = 7，其中Z与左侧输入维度相同
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
    parser = argparse.ArgumentParser(description='RINN模型训练脚本 - 6D输入预测1D输出')
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
        "num_epochs": 20,
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
experiment_name = f"rinn_6D_to_1D_{timestamp}"
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
print(f'Y数据形状: {y_data.shape}')  # (频率点数, 样本数)

# 数据重构：将数据转换为(样本数*频率点数, 6)的输入和(样本数*频率点数, 1)的输出
# 输入: [几何参数1, 几何参数2, 几何参数3, 几何参数4, 几何参数5, 频率]
# 输出: [S11值]

n_samples = len(x_samples)
n_freq = len(freq_data)

# 构建新的输入和输出数据
new_x = []
new_y = []

for i in range(n_samples):
    for j in range(n_freq):
        # 输入: 5维几何参数 + 1维频率
        input_data = np.concatenate([x_features[i], [freq_data[j]]])
        new_x.append(input_data)
        # 输出: 1维S11值
        new_y.append([y_data[j, i]])

new_x = np.array(new_x, dtype=np.float32)
new_y = np.array(new_y, dtype=np.float32)

print(f'重构后输入形状: {new_x.shape}')  # (样本数*频率点数, 6)
print(f'重构后输出形状: {new_y.shape}')  # (样本数*频率点数, 1)
print(f'输入示例 (第一个样本): {new_x[0]}')
print(f'输出示例 (第一个样本): {new_y[0]}')

# 数据标准化
normalization_method = config['data_params']['normalization_method']  # 'standard' 或 'robust'

# 先划分训练集和验证集的索引，再分别进行归一化
n_total = len(new_x)
indices = np.random.permutation(n_total)
train_size = int(0.8 * n_total)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 提取训练集数据
train_x = new_x[train_indices]
train_y = new_y[train_indices]

# 提取验证集数据
val_x = new_x[val_indices]
val_y = new_y[val_indices]

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
x_features_normalized = np.zeros_like(new_x, dtype=np.float32)
x_features_normalized[train_indices] = train_x_normalized
x_features_normalized[val_indices] = val_x_normalized

y_features_normalized = np.zeros_like(new_y, dtype=np.float32)
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
x_dim = new_x.shape[1]  # X维度：6（5维几何参数 + 1维频率）
y_dim = new_y.shape[1]       # Y维度：1
z_dim = x_dim                 # Z维度：6（与X维度相同）

# 左侧输入：X + 零填充 → 总维度 = x_dim + padding_dim = x_dim + y_dim = 7
padding_dim = y_dim
left_input_dim = x_dim + padding_dim

# 右侧输入：Y + Z → 总维度 = y_dim + z_dim = 7
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
left_val_input = np.concatenate((val_x_normalized, np.zeros((n_total - train_size, padding_dim), dtype=np.float32)), axis=1)

# 右侧输入：Y + Z（Z是随机生成的标准高斯分布）
train_z = np.random.randn(train_size, z_dim).astype(np.float32)
val_z = np.random.randn(n_total - train_size, z_dim).astype(np.float32)

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

# 导入模型
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss

# 创建模型
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

# 打印模型结构
def print_model_structure(model, indent=0):
    prefix = '  ' * indent
    print(f'{prefix}{model.__class__.__name__}')
    for name, child in model.named_children():
        print(f'{prefix}  {name}:')
        print_model_structure(child, indent + 2)

print('模型结构:')
print_model_structure(model)

# 计算模型参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f'模型参数总数: {total_params}')

# ============== 训练配置 ==============
print('\n=== 训练配置 ===')

skip_training = False  # 设置为True跳过训练，False执行完整训练

if not skip_training:
    # 损失权重
    weight_y = config['training_params']['loss_weights']['weight_y']  # 从配置中获取weight_y，提高正向预测精度
    weight_x = config['training_params']['loss_weights']['weight_x']  # 从配置中获取weight_x，保持X重建损失权重不变
    weight_z = config['training_params']['loss_weights']['weight_z']  # 从配置中获取weight_z，平衡生成多样性和稳定性
    clip_value = config['training_params']['clip_value']  # 从配置中获取clip_value，防止梯度爆炸
    
    print(f'损失权重 - Y: {weight_y}, X: {weight_x}, Z: {weight_z}')
    print(f'梯度裁剪值: {clip_value}')
    
    # 优化器
    lr = config['training_params']['learning_rate']  # 从配置中获取学习率，使用更精细的学习率调整
    weight_decay = config['training_params']['weight_decay']  # 从配置中获取权重衰减，平衡正则化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用AdamW优化器
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6)  # 调整patience，平衡收敛速度和稳定性
    
    print(f'初始学习率: {lr}')
    print(f'权重衰减: {weight_decay}')
    
    # 训练轮数和早停设置
    num_epochs = config['training_params']['num_epochs']  # 从配置中获取训练轮数，让模型有更多时间拟合
    best_val_loss = float('inf')
    patience = 10  # 减少早停耐心，与训练轮数匹配
    patience_counter = 0
    
    # 梯度累积步数
    grad_accum_steps = config['training_params']['gradient_accumulation_steps']  # 从配置中获取梯度累积步数
    print(f'梯度累积步数: {grad_accum_steps}')
    
    # 训练损失记录
    train_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}
    val_losses = {'total': [], 'y_loss': [], 'x_loss': [], 'z_loss': []}

# ============== 保存训练配置 ==============
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

# ============== 损失函数计算 ==============
def calculate_loss(left_input, right_input):
        """
        计算模型损失
        """
        # 前向传播 - 只传递left_input作为模型输入
        output, log_det = model(left_input)
        
        # 分离输出为Y和Z部分
        output_y = output[:, :y_dim]
        output_z = output[:, y_dim:]
        
        # 分离真实Y和Z部分
        target_y = right_input[:, :y_dim]
        target_z = right_input[:, y_dim:]
        
        # 计算损失
        y_loss = nmse_loss(output_y, target_y)
        z_loss = mmd_loss(output_z, target_z)
        
        # 逆向传播重建X
        reversed_output, reversed_log_det = model.inverse(output)
        reconstructed_x = reversed_output[:, :x_dim]
        original_x = left_input[:, :x_dim]
        x_loss = nmse_loss(reconstructed_x, original_x)
        
        # 总损失
        total_loss = weight_y * y_loss + weight_x * x_loss + weight_z * z_loss
        
        return total_loss, y_loss, x_loss, z_loss

# ============== 训练循环 ==============
if not skip_training:
    print('\n=== 开始训练 ===')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        total_train_loss = 0
        total_y_loss = 0
        total_x_loss = 0
        total_z_loss = 0
        
        # 梯度累积
        optimizer.zero_grad()
        
        for i, (left_input, right_input) in enumerate(train_loader):
            left_input = left_input.to(device)
            right_input = right_input.to(device)
            
            # 计算损失
            loss, y_loss, x_loss, z_loss = calculate_loss(left_input, right_input)
            
            # 梯度累积
            loss = loss / grad_accum_steps
            loss.backward()
            
            # 每grad_accum_steps步更新一次参数
            if (i + 1) % grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * grad_accum_steps
            total_y_loss += y_loss.item()
            total_x_loss += x_loss.item()
            total_z_loss += z_loss.item()
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_y_loss = total_y_loss / len(train_loader)
        avg_x_loss = total_x_loss / len(train_loader)
        avg_z_loss = total_z_loss / len(train_loader)
        
        # 验证模式
        model.eval()
        total_val_loss = 0
        total_val_y_loss = 0
        total_val_x_loss = 0
        total_val_z_loss = 0
        
        with torch.no_grad():
            for left_input, right_input in val_loader:
                left_input = left_input.to(device)
                right_input = right_input.to(device)
                
                # 计算损失
                loss, y_loss, x_loss, z_loss = calculate_loss(left_input, right_input)
                
                total_val_loss += loss.item()
                total_val_y_loss += y_loss.item()
                total_val_x_loss += x_loss.item()
                total_val_z_loss += z_loss.item()
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_y_loss = total_val_y_loss / len(val_loader)
        avg_val_x_loss = total_val_x_loss / len(val_loader)
        avg_val_z_loss = total_val_z_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 记录损失
        train_losses['total'].append(avg_train_loss)
        train_losses['y_loss'].append(avg_y_loss)
        train_losses['x_loss'].append(avg_x_loss)
        train_losses['z_loss'].append(avg_z_loss)
        
        val_losses['total'].append(avg_val_loss)
        val_losses['y_loss'].append(avg_val_y_loss)
        val_losses['x_loss'].append(avg_val_x_loss)
        val_losses['z_loss'].append(avg_val_z_loss)
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.6f}')
            print(f'  Y Loss: {avg_y_loss:.6f}, X Loss: {avg_x_loss:.6f}, Z Loss: {avg_z_loss:.6f}')
            print(f'Val Loss: {avg_val_loss:.6f}')
            print(f'  Y Loss: {avg_val_y_loss:.6f}, X Loss: {avg_val_x_loss:.6f}, Z Loss: {avg_val_z_loss:.6f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f'\n保存最佳模型 (验证损失: {best_val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n早停触发，训练结束 (Epoch {epoch+1})')
                break
    
    # 训练时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f'\n训练完成! 总训练时间: {training_time:.2f} 秒')
    print(f'最佳验证损失: {best_val_loss:.6f}')

# ============== 加载最佳模型 ==============
print('\n=== 加载最佳模型 ===')
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
model.eval()

# ============== 验证集性能评估 ==============
print('\n=== 验证集性能评估 ===')
model.eval()
total_val_loss = 0
total_val_y_loss = 0
total_val_x_loss = 0
total_val_z_loss = 0

with torch.no_grad():
    for left_input, right_input in val_loader:
        left_input = left_input.to(device)
        right_input = right_input.to(device)
        
        # 计算损失
        loss, y_loss, x_loss, z_loss = calculate_loss(left_input, right_input)
        
        total_val_loss += loss.item()
        total_val_y_loss += y_loss.item()
        total_val_x_loss += x_loss.item()
        total_val_z_loss += z_loss.item()

# 计算平均验证损失
avg_val_loss = total_val_loss / len(val_loader)
avg_val_y_loss = total_val_y_loss / len(val_loader)
avg_val_x_loss = total_val_x_loss / len(val_loader)
avg_val_z_loss = total_val_z_loss / len(val_loader)

print(f'验证集总损失: {avg_val_loss:.6f}')
print(f'验证集Y损失 (NMSE): {avg_val_y_loss:.6f}')
print(f'验证集X损失 (NMSE): {avg_val_x_loss:.6f}')
print(f'验证集Z损失 (MMD): {avg_val_z_loss:.6f}')

# ============== 可视化训练曲线 ==============
print('\n=== 可视化训练曲线 ===')
plt.figure(figsize=(12, 8))

# 总损失曲线
plt.subplot(2, 2, 1)
plt.plot(train_losses['total'], label='Train')
plt.plot(val_losses['total'], label='Validation')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Y损失曲线
plt.subplot(2, 2, 2)
plt.plot(train_losses['y_loss'], label='Train')
plt.plot(val_losses['y_loss'], label='Validation')
plt.title('Y Loss (NMSE)')
plt.xlabel('Epoch')
plt.ylabel('NMSE')
plt.legend()

# X损失曲线
plt.subplot(2, 2, 3)
plt.plot(train_losses['x_loss'], label='Train')
plt.plot(val_losses['x_loss'], label='Validation')
plt.title('X Loss (NMSE)')
plt.xlabel('Epoch')
plt.ylabel('NMSE')
plt.legend()

# Z损失曲线
plt.subplot(2, 2, 4)
plt.plot(train_losses['z_loss'], label='Train')
plt.plot(val_losses['z_loss'], label='Validation')
plt.title('Z Loss (MMD)')
plt.xlabel('Epoch')
plt.ylabel('MMD')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, 'training_losses.png'))
print(f'训练曲线已保存到: {os.path.join(checkpoint_dir, "training_losses.png")}')

# ============== 固定X预测Y（6维输入预测1维输出） ==============
print('\n=== 固定X预测Y (6维输入预测1维输出) ===')

# 选择5个测试样本进行可视化
n_test_samples = 5
test_indices = np.random.choice(len(val_x), n_test_samples, replace=False)
test_x = val_x[test_indices]
test_y = val_y[test_indices]

# 对测试样本进行归一化
test_x_normalized = (test_x - x_mean) / (x_std + 1e-8)

# 预测Y
with torch.no_grad():
        for i, (x_input, true_y) in enumerate(zip(test_x_normalized, test_y)):
            # 准备左侧输入 (X + 零填充)
            left_input = np.concatenate([x_input, np.zeros(padding_dim)], axis=0)
            left_input = torch.FloatTensor(left_input).unsqueeze(0).to(device)
            
            # 前向传播预测Y
            output, _ = model(left_input)
            predicted_y_normalized = output[:, :y_dim].cpu().numpy()
            
            # 反归一化预测结果
            predicted_y = predicted_y_normalized * y_std + y_mean
            
            # 计算NMSE
            nmse = np.mean((predicted_y - true_y) ** 2) / (np.mean(true_y ** 2) + 1e-8)
            
            print(f'\n测试样本 {i+1}:')
            print(f'  输入 (几何参数+频率): {test_x[i]}')
            print(f'  真实输出 (S11): {true_y[0]}')
            print(f'  预测输出 (S11): {predicted_y[0][0]}')
            print(f'  NMSE: {nmse:.6f}')

# ============== 固定Y回推X（1维输出回推6维输入） ==============
print('\n=== 固定Y回推X (1维输出回推6维输入) ===')

# 选择5个测试样本进行可视化
with torch.no_grad():
        for i, (x_input, true_y) in enumerate(zip(test_x_normalized, test_y)):
            # 准备真实Y的归一化输入
            true_y_normalized = (true_y - y_mean) / (y_std + 1e-8)
            
            # 创建输入：Y + 随机Z
            z = torch.randn(1, z_dim).to(device)
            input_data = np.concatenate([true_y_normalized, z.cpu().numpy().squeeze()], axis=0)
            input_data = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            
            # 逆向预测
            output, _ = model.inverse(input_data)
            predicted_x_normalized = output[:, :x_dim].cpu().numpy()
            
            # 反归一化
            predicted_x = predicted_x_normalized * x_std + x_mean
            
            # 计算NMSE
            nmse = np.mean((predicted_x - test_x[i]) ** 2) / (np.mean(test_x[i] ** 2) + 1e-8)
            
            print(f'\n测试样本 {i+1}:')
            print(f'  真实输出 (S11): {true_y[0]}')
            print(f'  真实输入 (几何参数+频率): {test_x[i]}')
            print(f'  预测输入 (几何参数+频率): {predicted_x[0]}')
            print(f'  NMSE: {nmse:.6f}')

# ============== 多解生成测试（1维输出生成多个6维输入） ==============
print('\n=== 多解生成测试 ===')

# 选择一个固定的Y值
fixed_y = test_y[0]
fixed_y_normalized = (fixed_y - y_mean) / (y_std + 1e-8)

# 生成多个X解
n_samples = 100
generated_x = []
predicted_ys = []

with torch.no_grad():
        for i in range(n_samples):
            # 生成随机Z
            z = torch.randn(1, z_dim).to(device)
            input_data = np.concatenate([fixed_y_normalized, z.cpu().numpy().squeeze()], axis=0)
            input_data = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            
            # 逆向传播回推X
            output, _ = model.inverse(input_data)
            predicted_x_normalized = output[:, :x_dim].cpu().numpy()
            
            # 反归一化预测结果
            predicted_x = predicted_x_normalized * x_std + x_mean
            generated_x.append(predicted_x.squeeze())
            
            # 验证生成的X是否能正确预测回Y
            x_input_normalized = (predicted_x - x_mean) / (x_std + 1e-8)
            left_input = np.concatenate([x_input_normalized.squeeze(), np.zeros(padding_dim)], axis=0)
            left_input = torch.FloatTensor(left_input).unsqueeze(0).to(device)
            
            # 前向传播预测Y
            output, _ = model(left_input)
            predicted_y_normalized = output[:, :y_dim].cpu().numpy()
            predicted_y = predicted_y_normalized * y_std + y_mean
            predicted_ys.append(predicted_y.squeeze())

# 转换为numpy数组
generated_x = np.array(generated_x)
predicted_ys = np.array(predicted_ys)

# 计算X的多样性（各维度的标准差）
x_diversity = np.std(generated_x, axis=0)
print(f'\nX多样性 (各维度标准差): {x_diversity}')
print(f'平均X多样性: {np.mean(x_diversity):.6f}')

# 计算Y预测误差
print(f'\nY预测误差:')
print(f'  平均误差: {np.mean(np.abs(predicted_ys - fixed_y)):.6f}')
print(f'  均方误差: {np.mean((predicted_ys - fixed_y) ** 2):.6f}')

# 保存生成的X和预测的Y
np.save(os.path.join(checkpoint_dir, 'generated_xs.npy'), generated_x)
np.save(os.path.join(checkpoint_dir, 'predicted_ys.npy'), predicted_ys)
print(f'\n生成的X和预测的Y已保存到: {checkpoint_dir}')

print('\n=== 实验完成 ===')