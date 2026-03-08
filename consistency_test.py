import os
import torch
import numpy as np
from datetime import datetime

# ============== 配置区域 ==============
# 最佳模型所在目录
MODEL_DIR = 'model_checkpoints_rinn/rinn_correct_structure_20260307_224946'

# 输出结果目录（添加时间戳确保唯一性）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'model_checkpoints_rinn/consistency_test_{timestamp}'

# 批量大小
batch_size = 1
# ============== 配置区域结束 ==============

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

# 不加载训练权重，使用随机初始化的模型进行测试
# checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('使用随机初始化的模型进行一致性测试（未加载训练权重）')
# print(f'模型加载完成: {checkpoint_path}')
# print(f'最佳验证损失: {checkpoint["val_loss"]:.6f}, Epoch: {checkpoint["epoch"]+1}')

# ============== 一致性检测 ==============
print('\n=== 测试2：部分输入（零填充）的一致性 ===')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 生成部分输入（只有前5维有数据，其余为零）
x_dim = 5
padding_dim = left_input_dim - x_dim  # 使用配置中的左输入维度
partial_input = torch.randn(batch_size, x_dim).to(device)
padded_input = torch.cat([partial_input, torch.zeros(batch_size, padding_dim).to(device)], dim=1)

# 2. 正向变换
output, log_det_forward = model(padded_input)

# 3. 逆向变换
reconstructed, log_det_inverse = model.inverse(output)

# 4. 提取前5维，其余零填充
reconstructed_partial = reconstructed[:, :x_dim]
reconstructed_padded = torch.cat([reconstructed_partial, torch.zeros(batch_size, padding_dim).to(device)], dim=1)

# 5. 再次正向变换
output_from_reconstructed, _ = model(reconstructed_padded)

# 6. 验证一致性
consistency_error = torch.mean((output - output_from_reconstructed) ** 2).item()
print(f'部分输入形状: {partial_input.shape}')
print(f'填充后输入形状: {padded_input.shape}')
print(f'第一次输出形状: {output.shape}')
print(f'重建输入形状: {reconstructed.shape}')
print(f'重建部分形状: {reconstructed_partial.shape}')
print(f'再次填充后形状: {reconstructed_padded.shape}')
print(f'第二次输出形状: {output_from_reconstructed.shape}')
print(f'一致性误差 MSE: {consistency_error:.10f}')
if consistency_error < 1e-5:
    print('✓ 一致性验证通过！')
else:
    print('⚠ 一致性验证不通过')

# 7. 保存输入和重建的值到文件

# 保存输入的x值（前5维）
input_x = partial_input.cpu().numpy()
np.savetxt(os.path.join(OUTPUT_DIR, 'input_x.txt'), input_x, fmt='%.8f', delimiter=',')

# 保存完整的输入（包括零填充）
input_padded = padded_input.cpu().numpy()
np.savetxt(os.path.join(OUTPUT_DIR, 'input_padded.txt'), input_padded, fmt='%.8f', delimiter=',')

# 保存逆向变换得到的x值（完整202维）
reconstructed_x = reconstructed.detach().cpu().numpy()
np.savetxt(os.path.join(OUTPUT_DIR, 'reconstructed_x.txt'), reconstructed_x, fmt='%.8f', delimiter=',')

# 保存逆向变换得到的x值（前5维）
reconstructed_x_partial = reconstructed_partial.detach().cpu().numpy()
np.savetxt(os.path.join(OUTPUT_DIR, 'reconstructed_x_partial.txt'), reconstructed_x_partial, fmt='%.8f', delimiter=',')

# 保存两个y值到txt文件
np.savetxt(os.path.join(OUTPUT_DIR, 'test2_output.txt'), output.detach().cpu().numpy(), fmt='%.8f', delimiter=',')
np.savetxt(os.path.join(OUTPUT_DIR, 'test2_output_from_reconstructed.txt'), output_from_reconstructed.detach().cpu().numpy(), fmt='%.8f', delimiter=',')

print(f'\n输入和重建的值已保存到 {OUTPUT_DIR} 目录')
print(f'input_x.txt: 输入的前5维值')
print(f'input_padded.txt: 完整的输入（包括零填充）')
print(f'reconstructed_x.txt: 逆向变换得到的完整202维值')
print(f'reconstructed_x_partial.txt: 逆向变换得到的前5维值')
print(f'test2_output.txt: 第一次正向变换的输出')
print(f'test2_output_from_reconstructed.txt: 第二次正向变换的输出')
