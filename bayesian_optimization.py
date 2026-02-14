"""
贝叶斯优化脚本，用于优化RINN模型的超参数
"""
import os
import torch
import numpy as np
import json
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# 确保optuna模块已安装
try:
    import optuna
except ImportError:
    print("正在安装optuna模块...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

print(f"使用optuna版本: {optuna.__version__}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f'使用设备: {device}')

# ============== 数据加载与预处理 ==============
print('\n=== 加载数据 ===')

# 提取几何参数信息
def extract_geometry_params(col_name):
    """从列名中提取几何参数H1, H2, H3, H_C1, H_C2"""
    import re
    col_name = col_name.replace('\n', '').replace('"', '')
    
    h1_match = re.search(r"H1='([\d.]+)mm'", col_name)
    h2_match = re.search(r"H2='([\d.]+)mm'", col_name)
    h3_match = re.search(r"H3='([\d.]+)mm'", col_name)
    hc1_match = re.search(r"H_C1='([\d.]+)mm'", col_name)
    hc2_match = re.search(r"H_C2='([\d.]+)mm'", col_name)
    
    # 检查是否是实部还是虚部
    is_real = 're(S(1,1))' in col_name
    is_imag = 'im(S(1,1))' in col_name
    
    if all([h1_match, h2_match, h3_match, hc1_match, hc2_match]):
        return {
            'params': [
                float(h1_match.group(1)),
                float(h2_match.group(1)),
                float(h3_match.group(1)),
                float(hc1_match.group(1)),
                float(hc2_match.group(1))
            ],
            'type': 'real' if is_real else 'imaginary'
        }
    return None

def load_data_from_csv(data_path):
    """从CSV文件加载数据"""
    print(f'正在加载数据文件: {data_path}')
    
    # 读取表头获取几何参数
    import csv
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    # 提取所有几何参数样本和数据列
    geometry_dict = {}
    
    for i, col in enumerate(header[1:]):  # 跳过第一列频率
        params = extract_geometry_params(col)
        if params:
            geo_key = tuple(params['params'])
            if geo_key not in geometry_dict:
                geometry_dict[geo_key] = {'real': None, 'imag': None}
            if params['type'] == 'real':
                geometry_dict[geo_key]['real'] = i+1
            else:
                geometry_dict[geo_key]['imag'] = i+1
    
    # 只保留同时有实部和虚部的样本
    valid_samples = []
    real_columns = []
    imag_columns = []
    
    for geo_key, cols in geometry_dict.items():
        if cols['real'] is not None and cols['imag'] is not None:
            valid_samples.append(list(geo_key))
            real_columns.append(cols['real'])
            imag_columns.append(cols['imag'])
    
    x_features = np.array(valid_samples, dtype=np.float32)
    print(f'  X特征形状: {x_features.shape}')
    print(f'  X特征示例 (第一个样本): {x_features[0]}')
    
    # 读取S11数据
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    freq_data = data[:, 0]  # 频率数据
    print(f'  频率点数: {len(freq_data)}')
    print(f'  频率范围: {freq_data[0]} GHz - {freq_data[-1]} GHz')
    
    # 提取实部和虚部数据列
    real_data = data[:, real_columns]  # 实部数据
    imag_data = data[:, imag_columns]  # 虚部数据
    
    # 转置为(样本数, 频率点数)
    real_data = real_data.T
    imag_data = imag_data.T
    
    # 合并实部和虚部为一个202维的输出（101维实部 + 101维虚部）
    y_data = np.concatenate((real_data, imag_data), axis=1)
    print(f'  Y数据形状: {y_data.shape} (101维实部 + 101维虚部)')
    
    return x_features, y_data, freq_data

# 加载训练数据
train_data_path = 'data/S Parameter Plot300.csv'
train_x, train_y, freq_data = load_data_from_csv(train_data_path)

# 加载验证数据
val_data_path = 'data/S Parameter Plot200.csv'
val_x, val_y, _ = load_data_from_csv(val_data_path)

print(f'\n训练集样本数: {len(train_x)}')
print(f'验证集样本数: {len(val_x)}')

# 数据标准化
def normalize_data(train_x, val_x, train_y, val_y, method='robust'):
    """数据标准化"""
    if method == 'standard':
        # 使用均值-标准差归一化
        x_mean = train_x.mean(axis=0)
        x_std = train_x.std(axis=0)
        train_x_normalized = (train_x - x_mean) / (x_std + 1e-8)
        val_x_normalized = (val_x - x_mean) / (x_std + 1e-8)
        
        y_mean = train_y.mean(axis=0)
        y_std = train_y.std(axis=0)
        
        # 确保y_mean和y_std是202维（101维实部 + 101维虚部）
        y_mean = y_mean[:202]
        y_std = y_std[:202]
        
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
        
        # 确保y_mean和y_std是202维（101维实部 + 101维虚部）
        y_mean = y_mean[:202]
        y_std = y_std[:202]
    
    return train_x_normalized, val_x_normalized, train_y_normalized, val_y_normalized, x_mean, x_std, y_mean, y_std

# 归一化数据
train_x_normalized, val_x_normalized, train_y_normalized, val_y_normalized, x_mean, x_std, y_mean, y_std = normalize_data(
    train_x, val_x, train_y, val_y, method='robust'
)

# ============== 维度处理 ==============
print('\n=== 维度处理与数据结构 ===')

# 配置参数
x_dim = train_x.shape[1]  # X维度：5
y_dim = train_y.shape[1]  # Y维度：202（101维实部 + 101维虚部）
z_dim = x_dim             # Z维度：5（与X维度相同）

# 左侧输入：X + 零填充 → 总维度 = x_dim + padding_dim = 5 + 202 = 207
padding_dim = y_dim
left_input_dim = x_dim + padding_dim

# 右侧输入：Y + Z → 总维度 = y_dim + z_dim = 202 + 5 = 207
right_input_dim = y_dim + z_dim

print(f'X维度: {x_dim}, Y维度: {y_dim}, Z维度: {z_dim}')
print(f'左侧输入维度: {left_input_dim} (X: {x_dim} + 零填充: {padding_dim})')
print(f'右侧输入维度: {right_input_dim} (Y: {y_dim} + Z: {z_dim})')
print(f'总输入/输出维度: {left_input_dim} (左右侧维度相同)')

# 创建训练集和验证集数据
# 左侧输入：X + 零填充
left_train_input = np.concatenate((train_x_normalized, np.zeros((len(train_x_normalized), padding_dim), dtype=np.float32)), axis=1)
left_val_input = np.concatenate((val_x_normalized, np.zeros((len(val_x_normalized), padding_dim), dtype=np.float32)), axis=1)

# 注意：Z现在在每个epoch重新采样，不再在这里固定生成
# 右侧输入将在训练循环中动态生成

# 转换为torch张量（左侧输入固定）
left_train = torch.FloatTensor(left_train_input)
left_val = torch.FloatTensor(left_val_input)

print('\n数据集划分:')
print(f'  训练集: {len(left_train)} 样本')
print(f'  验证集: {len(left_val)} 样本')
print('  注意：Z将在每个epoch重新采样，增强模型泛化能力')

# ============== 模型定义与损失函数 ==============
from R_INN_model.rinn_model import RINNModel
from R_INN_model.loss_methods import mmd_loss, nmse_loss, weighted_nmse_loss

def calculate_loss(model, left_input, right_input, x_dim, y_dim, weight_y, weight_x, weight_z, device):
    """计算损失"""
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
    
    # 使用MMD损失计算x损失
    x_loss = mmd_loss(real_x, reconstructed_x)
    
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

# ============== 训练函数 ==============
def train_model(model, left_train, left_val, train_y_normalized, val_y_normalized, 
                batch_size, optimizer, scheduler, num_epochs, 
                x_dim, y_dim, z_dim, weight_y, weight_x, weight_z, device, patience=30):
    """训练模型 - 每个epoch重新采样Z"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # ========== 每个epoch重新采样Z ==========
        # 重新采样训练集和验证集的Z
        train_z = np.random.randn(len(train_y_normalized), z_dim).astype(np.float32)
        val_z = np.random.randn(len(val_y_normalized), z_dim).astype(np.float32)
        
        # 创建右侧输入：Y + Z
        right_train_input = np.concatenate((train_y_normalized, train_z), axis=1)
        right_val_input = np.concatenate((val_y_normalized, val_z), axis=1)
        
        # 转换为torch张量
        right_train = torch.FloatTensor(right_train_input)
        right_val = torch.FloatTensor(right_val_input)
        
        # 创建DataLoader
        train_dataset = TensorDataset(left_train, right_train)
        val_dataset = TensorDataset(left_val, right_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            left_batch = batch[0].to(device)
            right_batch = batch[1].to(device)
            
            # 计算损失
            losses = calculate_loss(model, left_batch, right_batch, x_dim, y_dim, 
                                   weight_y, weight_x, weight_z, device)
            
            # 反向传播
            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += losses["total_loss"].item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                left_batch = batch[0].to(device)
                right_batch = batch[1].to(device)
                
                # 计算损失
                losses = calculate_loss(model, left_batch, right_batch, x_dim, y_dim, 
                                       weight_y, weight_x, weight_z, device)
                
                val_loss += losses["total_loss"].item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停触发! 验证损失连续{patience}个epoch没有改善')
                break
        
        # 每10个epoch打印一次
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return best_val_loss

# ============== 目标函数 ==============
def objective(trial):
    """优化目标函数"""
    # 超参数搜索空间
    params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128, step=8),
        "num_blocks": trial.suggest_int("num_blocks", 3, 8),
        "num_stages": trial.suggest_int("num_stages", 1, 4),
        "num_cycles_per_stage": trial.suggest_int("num_cycles_per_stage", 1, 3),
        "ratio_toZ_after_flowstage": trial.suggest_float("ratio_toZ_after_flowstage", 0.1, 0.7),
        "ratio_x1_x2_inAffine": trial.suggest_float("ratio_x1_x2_inAffine", 0.05, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "weight_y": trial.suggest_float("weight_y", 0.5, 2.0),
        "weight_x": trial.suggest_float("weight_x", 0.1, 1.0),
        "weight_z": trial.suggest_float("weight_z", 0.1, 0.5)
    }
    
    print(f"\n尝试参数: {params}")
    
    # 获取batch_size
    batch_size = params["batch_size"]
    
    # 创建模型
    model = RINNModel(
        input_dim=left_input_dim,
        hidden_dim=params["hidden_dim"],
        num_blocks=params["num_blocks"],
        num_stages=params["num_stages"],
        num_cycles_per_stage=params["num_cycles_per_stage"],
        ratio_toZ_after_flowstage=params["ratio_toZ_after_flowstage"],
        ratio_x1_x2_inAffine=params["ratio_x1_x2_inAffine"]
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=1e-6
    )
    
    # 训练模型
    try:
        best_val_loss = train_model(
            model, left_train, left_val, train_y_normalized, val_y_normalized,
            batch_size, optimizer, scheduler,
            num_epochs=50,  # 为了加快优化速度，使用较少的epoch
            x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
            weight_y=params["weight_y"],
            weight_x=params["weight_x"],
            weight_z=params["weight_z"],
            device=device,
            patience=20
        )
        print(f"最佳验证损失: {best_val_loss:.6f}")
        return best_val_loss
    except Exception as e:
        print(f"训练失败: {e}")
        return float('inf')

# ============== 主函数 ==============
def main():
    # 创建优化器
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,  # 初始随机搜索次数
            seed=42
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # 运行优化
    n_trials = 5  # 快速测试：5次尝试
    print(f"\n开始贝叶斯优化，计划运行 {n_trials} 次尝试...")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('optimization_results', f'rinn_bayesian_opt_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== 最佳参数 ===")
    print(f"最佳验证损失: {best_value:.6f}")
    print(f"最佳参数: {best_params}")
    
    # 保存最佳参数
    with open(os.path.join(results_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    
    # 保存所有尝试的参数和结果
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        })
    
    with open(os.path.join(results_dir, 'all_trials.json'), 'w', encoding='utf-8') as f:
        json.dump(trials_data, f, ensure_ascii=False, indent=2)
    
    # 保存优化历史
    with open(os.path.join(results_dir, 'optimization_history.txt'), 'w') as f:
        f.write(f"贝叶斯优化结果\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"尝试次数: {n_trials}\n")
        f.write(f"最佳验证损失: {best_value:.6f}\n")
        f.write(f"最佳参数: {json.dumps(best_params, ensure_ascii=False, indent=2)}\n")
    
    print(f"\n优化结果已保存到: {results_dir}")
    
    # 可视化优化过程
    try:
        import matplotlib.pyplot as plt
        # 设置中文字体支持 - 尝试多种字体，提高兼容性
        plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 绘制验证损失随尝试次数的变化
        plt.figure(figsize=(12, 6))
        values = [trial.value for trial in study.trials if trial.value is not None]
        plt.plot(range(len(values)), values, 'b-', alpha=0.5)
        plt.scatter(range(len(values)), values, c='b', s=10)
        plt.xlabel('尝试次数')
        plt.ylabel('验证损失')
        plt.title('贝叶斯优化过程')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("优化历史图已保存")
        
        # 绘制参数重要性
        importances = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), list(importances.values()), align='center')
        plt.xticks(range(len(importances)), list(importances.keys()), rotation=45, ha='right')
        plt.xlabel('参数')
        plt.ylabel('重要性')
        plt.title('参数重要性分析')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'param_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("参数重要性图已保存")
    except Exception as e:
        print(f"绘制图表失败: {e}")

if __name__ == "__main__":
    # 检查是否安装了optuna
    try:
        import optuna
        print(f"Optuna版本: {optuna.__version__}")
    except ImportError:
        print("正在安装Optuna...")
        import subprocess
        subprocess.run(["pip", "install", "optuna"], check=True)
        import optuna
        print(f"Optuna版本: {optuna.__version__}")
    
    # 检查是否安装了matplotlib
    try:
        import matplotlib
        print(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("正在安装Matplotlib...")
        import subprocess
        subprocess.run(["pip", "install", "matplotlib"], check=True)
    
    # 运行主函数
    main()
