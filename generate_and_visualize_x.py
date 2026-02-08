import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# ============== 个人配置区域（直接在这里修改） ==============
# 输入的y文件路径（包含实部和虚部数据）
Y_FILE_PATH = 'data/S Parameter Plot1perfect.csv'

# 生成的x样本数量（对z取样次数）
NUM_SAMPLES = 3

# 最佳模型所在目录
MODEL_DIR = 'model_checkpoints_rinn/rinn_correct_structure_20260208_104358'

# 输出结果目录
OUTPUT_DIR = 'model_checkpoints_rinn/generate_x_results'

# 标准X样本（用于误差分析）
STANDARD_X = np.array([3.757, 4.552, 4.125, 3.114, 2.893])

# 随机种子（None表示不固定，每次运行结果不同）
RANDOM_SEED = 42  # 例如：42 表示固定随机种子，结果可复现
# ============== 个人配置区域结束 ==============

# 参数名称
PARAM_NAMES = ['H1[mm]', 'H2[mm]', 'H3[mm]', 'H_C1[mm]', 'H_C2[mm]']

# 设置随机种子
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f'使用固定随机种子: {RANDOM_SEED}')
else:
    print('不使用固定随机种子，每次运行结果可能不同')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ============== 加载模型配置 ==============
print('\n=== 加载模型配置 ===')

# 加载训练配置
config_path = os.path.join(MODEL_DIR, 'training_config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    training_config = eval(f.read())

# 提取关键参数
model_config = training_config['model_config']
data_info = training_config['data_info']

input_dim = model_config['input_dim']
hidden_dim = model_config['hidden_dim']
num_blocks = model_config['num_blocks']
num_stages = model_config['num_stages']
num_cycles_per_stage = model_config['num_cycles_per_stage']
ratio_toZ_after_flowstage = model_config['ratio_toZ_after_flowstage']
ratio_x1_x2_inAffine = model_config['ratio_x1_x2_inAffine']

x_dim = data_info['x_dim']
y_dim = data_info['y_dim']
z_dim = data_info['z_dim']
left_input_dim = data_info['left_input_dim']
right_input_dim = data_info['right_input_dim']

# 加载归一化参数
x_mean = np.array(data_info['x_mean'])
x_std = np.array(data_info['x_std'])
y_mean = np.array(data_info['y_mean'])
y_std = np.array(data_info['y_std'])

print(f'模型输入维度: {input_dim}')
print(f'X维度: {x_dim}, Y维度: {y_dim}, Z维度: {z_dim}')

# ============== 加载模型 ==============
print('\n=== 加载模型 ===')

# 导入RINN模型
from R_INN_model.rinn_model import RINNModel

# 创建模型
model = RINNModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks,
    num_stages=num_stages,
    num_cycles_per_stage=num_cycles_per_stage,
    ratio_toZ_after_flowstage=ratio_toZ_after_flowstage,
    ratio_x1_x2_inAffine=ratio_x1_x2_inAffine
).to(device)

# 加载最佳模型权重
checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f'模型加载完成: {checkpoint_path}')
print(f'最佳验证损失: {checkpoint["val_loss"]:.6f}, Epoch: {checkpoint["epoch"]+1}')

# ============== 加载Y数据 ==============
print('\n=== 加载Y数据 ===')

data = pd.read_csv(Y_FILE_PATH)
freq_data = data.iloc[:, 0].values
real_data = data.iloc[:, 1].values
imag_data = data.iloc[:, 2].values
y_data = np.concatenate((real_data, imag_data))

print(f'频率范围: {freq_data[0]} GHz - {freq_data[-1]} GHz')
print(f'Y数据维度: {y_data.shape[0]}')

# 确保y_data维度匹配归一化参数的维度
norm_dim = len(y_mean)
if y_data.shape[0] != norm_dim:
    print(f'警告: Y数据维度 ({y_data.shape[0]}) 与归一化参数维度 ({norm_dim}) 不匹配')
    if y_data.shape[0] < norm_dim:
        padding = np.zeros(norm_dim - y_data.shape[0])
        y_data = np.concatenate((y_data, padding))
        print(f'已零填充到 {norm_dim} 维')
    else:
        y_data = y_data[:norm_dim]
        print(f'已截断到 {norm_dim} 维')

y_normalized = (y_data - y_mean) / (y_std + 1e-8)

# ============== 生成X样本 ==============
print(f'\n=== 生成 {NUM_SAMPLES} 个X样本 ===')

z_samples = np.random.randn(NUM_SAMPLES, z_dim).astype(np.float32)
y_repeated = np.repeat(y_normalized.reshape(1, -1), NUM_SAMPLES, axis=0)
y_padding_dim = right_input_dim - len(y_normalized) - z_dim
y_padding = np.zeros((NUM_SAMPLES, y_padding_dim), dtype=np.float32)
right_inputs = np.concatenate((y_repeated, z_samples, y_padding), axis=1)
right_inputs_tensor = torch.FloatTensor(right_inputs).to(device)

with torch.no_grad():
    reconstructed_lefts, _ = model.inverse(right_inputs_tensor)
    reconstructed_xs_normalized = reconstructed_lefts[:, :x_dim]
    reconstructed_xs = reconstructed_xs_normalized.cpu().numpy() * x_std + x_mean

print(f'生成的X样本形状: {reconstructed_xs.shape}')

# ============== 计算误差和验证 ==============
print('\n=== 误差分析和验证 ===')

absolute_errors = []
relative_errors = []
validation_results = []

for i, sample in enumerate(reconstructed_xs):
    abs_err = np.abs(sample - STANDARD_X)
    rel_err = abs_err / (STANDARD_X + 1e-8) * 100
    absolute_errors.append(abs_err)
    relative_errors.append(rel_err)
    
    # 验证生成结果
    x_test = sample.reshape(1, -1)
    x_test_normalized = (x_test - x_mean) / (x_std + 1e-8)
    padding_dim = left_input_dim - x_dim
    left_test_input = np.concatenate((x_test_normalized, np.zeros((1, padding_dim), dtype=np.float32)), axis=1)
    left_test_input = torch.FloatTensor(left_test_input).to(device)
    
    with torch.no_grad():
        predicted_right, _, _ = model(left_test_input, return_intermediate=True)
        predicted_y_normalized = predicted_right[:, :len(y_normalized)]
        predicted_y = predicted_y_normalized.cpu().numpy() * y_std + y_mean
    
    y_diff = np.abs(predicted_y[0] - y_data)
    y_mse = np.mean(y_diff ** 2)
    validation_results.append(y_mse)
    
    print(f'\n样本 {i+1}:')
    print(f'  H1={sample[0]:.4f}mm, H2={sample[1]:.4f}mm, H3={sample[2]:.4f}mm')
    print(f'  H_C1={sample[3]:.4f}mm, H_C2={sample[4]:.4f}mm')
    print(f'  平均相对误差: {np.mean(rel_err):.2f}%')
    print(f'  验证MSE: {y_mse:.6f}')

absolute_errors = np.array(absolute_errors)
relative_errors = np.array(relative_errors)

# ============== 统计分析 ==============
print('\n=== 统计分析 ===')

param_avg_abs_err = np.mean(absolute_errors, axis=0)
param_avg_rel_err = np.mean(relative_errors, axis=0)
sample_avg_rel_err = np.mean(relative_errors, axis=1)
best_sample_idx = np.argmin(sample_avg_rel_err)

print('\n各参数平均误差:')
for i, (param, abs_err, rel_err) in enumerate(zip(PARAM_NAMES, param_avg_abs_err, param_avg_rel_err)):
    print(f'  {param}: 绝对误差={abs_err:.4f}mm, 相对误差={rel_err:.2f}%')

print(f'\n最优样本: 样本 {best_sample_idx+1}，平均相对误差={sample_avg_rel_err[best_sample_idx]:.2f}%')

# ============== 可视化 ==============
print('\n=== 生成可视化 ===')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 对比柱状图
plt.figure(figsize=(15, 8))
grid = np.arange(len(PARAM_NAMES))
width = 0.15
plt.bar(grid - 2*width, STANDARD_X, width, label='Standard Sample', color='blue', alpha=0.8)
colors = ['green', 'orange', 'red']
for i in range(len(reconstructed_xs)):
    plt.bar(grid + (i-1)*width, reconstructed_xs[i], width, label=f'Generated Sample {i+1}', color=colors[i], alpha=0.6)
plt.xlabel('Parameter')
plt.ylabel('Value (mm)')
plt.title('Comparison of Standard X Sample and Generated X Samples')
plt.xticks(grid, PARAM_NAMES)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'x_samples_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2. 误差分析图
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(relative_errors, cmap='RdYlGn_r', vmin=0, vmax=10)
plt.colorbar(label='Relative Error (%)')
plt.xticks(np.arange(len(PARAM_NAMES)), PARAM_NAMES, rotation=45, ha='right')
plt.yticks(np.arange(len(reconstructed_xs)), [f'Sample {i+1}' for i in range(len(reconstructed_xs))])
plt.title('Relative Error Heatmap')
plt.subplot(1, 2, 2)
plt.boxplot(relative_errors.T)
plt.ylabel('Relative Error (%)')
plt.title('Error Distribution by Parameter')
plt.xticks(np.arange(1, len(PARAM_NAMES)+1), PARAM_NAMES, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'x_error_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. 雷达图
plt.figure(figsize=(12, 8))
ax = plt.subplot(111, polar=True)
N = len(PARAM_NAMES)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
all_x = np.vstack([STANDARD_X, reconstructed_xs])
x_min = all_x.min(axis=0)
x_max = all_x.max(axis=0)
std_normalized = (STANDARD_X - x_min) / (x_max - x_min + 1e-8)
std_normalized = np.append(std_normalized, std_normalized[0])
ax.plot(angles, std_normalized, 'b-', linewidth=2, label='Standard Sample', marker='o')
ax.fill(angles, std_normalized, 'b', alpha=0.1)
colors = ['g', 'y', 'r']
for i in range(len(reconstructed_xs)):
    gen_normalized = (reconstructed_xs[i] - x_min) / (x_max - x_min + 1e-8)
    gen_normalized = np.append(gen_normalized, gen_normalized[0])
    ax.plot(angles, gen_normalized, f'{colors[i]}-', linewidth=1.5, label=f'Generated Sample {i+1}', marker='s')
    ax.fill(angles, gen_normalized, colors[i], alpha=0.05)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(PARAM_NAMES)
plt.title('Radar Chart Comparison of X Samples', size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'x_samples_radar.png'), dpi=150, bbox_inches='tight')
plt.close()

print('可视化图表已保存')

# ============== 保存结果 ==============
print('\n=== 保存结果 ===')

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 保存CSV文件
output_file = os.path.join(OUTPUT_DIR, 'generated_x_samples.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    f.write(','.join(PARAM_NAMES) + '\n')
    for sample in reconstructed_xs:
        f.write(','.join(map(str, sample)) + '\n')
    f.write('\n=== 统计分析 ===\n\n')
    f.write('各参数平均误差:\n')
    for i, (param, abs_err, rel_err) in enumerate(zip(PARAM_NAMES, param_avg_abs_err, param_avg_rel_err)):
        f.write(f'{param}: 绝对误差={abs_err:.4f}mm, 相对误差={rel_err:.2f}%\n')
    f.write('\n各样本平均相对误差:\n')
    for i, err in enumerate(sample_avg_rel_err):
        f.write(f'样本 {i+1}: {err:.2f}%\n')
    f.write(f'\n最优样本: 样本 {best_sample_idx+1}，平均相对误差={sample_avg_rel_err[best_sample_idx]:.2f}%\n')
    f.write(f'最优样本参数: {reconstructed_xs[best_sample_idx]}\n')
    f.write('\n=== 生成时间 ===\n')
    f.write(f'生成时间: {current_time}\n')

# 保存JSON文件
detailed_results = {
    'input_file': Y_FILE_PATH,
    'model_dir': MODEL_DIR,
    'num_samples': NUM_SAMPLES,
    'random_seed': RANDOM_SEED,
    'generation_time': current_time,
    'generated_x_samples': reconstructed_xs.tolist(),
    'z_samples': z_samples.tolist(),
    'standard_x': STANDARD_X.tolist(),
    'error_analysis': {
        'param_avg_abs_err': param_avg_abs_err.tolist(),
        'param_avg_rel_err': param_avg_rel_err.tolist(),
        'sample_avg_rel_err': sample_avg_rel_err.tolist(),
        'best_sample_idx': int(best_sample_idx),
        'best_sample_err': float(sample_avg_rel_err[best_sample_idx])
    },
    'validation_results': validation_results
}

with open(os.path.join(OUTPUT_DIR, 'generation_details.json'), 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, ensure_ascii=False, indent=2)

print(f'结果已保存到: {OUTPUT_DIR}')
print(f'生成时间: {current_time}')