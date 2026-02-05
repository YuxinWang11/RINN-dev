"""
分析 S Parameter Plot 7.csv 文件结构
理解数据格式和参数提取方法
"""
import csv
import numpy as np
import re

# 读取CSV文件
data_path = 'data/S Parameter Plot testdata8.csv'

print("=" * 80)
print("CSV文件结构分析")
print("=" * 80)

# 读取表头
with open(data_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)

print(f"\n总列数: {len(header)}")
print(f"\n第一列（频率列）: {header[0]}")

# 分析数据列（跳过第一列频率）
print(f"\n数据列数量: {len(header) - 1}")

# 提取几何参数的函数
def extract_geometry_params(col_name):
    """从列名中提取几何参数H1, H2, H3, H_C1, H_C2"""
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
            'H1': float(h1_match.group(1)),
            'H2': float(h2_match.group(1)),
            'H3': float(h3_match.group(1)),
            'H_C1': float(hc1_match.group(1)),
            'H_C2': float(hc2_match.group(1)),
            'type': 'real' if is_real else 'imaginary'
        }
    return None

# 提取所有样本的几何参数
samples = []
real_columns = []
imag_columns = []

for i, col in enumerate(header[1:], start=1):  # 跳过第一列频率
    params = extract_geometry_params(col)
    if params:
        samples.append(params)
        if params['type'] == 'real':
            real_columns.append(i)
        else:
            imag_columns.append(i)

print(f"\n成功提取的样本数: {len(samples)}")
print(f"实部列数量: {len(real_columns)}")
print(f"虚部列数量: {len(imag_columns)}")

# 显示前几个样本
print("\n" + "=" * 80)
print("前5个样本的几何参数:")
print("=" * 80)
for i, sample in enumerate(samples[:5]):
    print(f"\n样本 {i+1}:")
    print(f"  H1: {sample['H1']:.5f} mm")
    print(f"  H2: {sample['H2']:.5f} mm")
    print(f"  H3: {sample['H3']:.5f} mm")
    print(f"  H_C1: {sample['H_C1']:.5f} mm")
    print(f"  H_C2: {sample['H_C2']:.5f} mm")
    print(f"  类型: {sample['type']}")

# 读取数据
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
print("\n" + "=" * 80)
print("数据形状分析:")
print("=" * 80)
print(f"数据矩阵形状: {data.shape}")
print(f"  - 行数（频率点数）: {data.shape[0]}")
print(f"  - 列数（频率 + 数据列）: {data.shape[1]}")

# 提取频率数据
freq_data = data[:, 0]
print(f"\n频率范围: {freq_data[0]:.6f} GHz - {freq_data[-1]:.6f} GHz")
print(f"频率点数: {len(freq_data)}")

# 提取实部和虚部数据
real_data = data[:, real_columns]
imag_data = data[:, imag_columns]

print(f"\n实部数据形状: {real_data.shape}")
print(f"虚部数据形状: {imag_data.shape}")

# 检查实部和虚部是否成对出现
print(f"\n实部和虚部是否成对: {len(real_columns) == len(imag_columns)}")

# 检查数据质量
print("\n" + "=" * 80)
print("数据质量检查:")
print("=" * 80)
print(f"实部数据 - 最小值: {real_data.min():.6f}, 最大值: {real_data.max():.6f}")
print(f"虚部数据 - 最小值: {imag_data.min():.6f}, 最大值: {imag_data.max():.6f}")
print(f"实部数据 - 是否包含NaN: {np.isnan(real_data).any()}")
print(f"虚部数据 - 是否包含NaN: {np.isnan(imag_data).any()}")
print(f"实部数据 - 是否包含无穷大: {np.isinf(real_data).any()}")
print(f"虚部数据 - 是否包含无穷大: {np.isinf(imag_data).any()}")

# 分析几何参数的唯一组合
print("\n" + "=" * 80)
print("几何参数组合分析:")
print("=" * 80)
geometry_params = []
for sample in samples:
    params = (sample['H1'], sample['H2'], sample['H3'], sample['H_C1'], sample['H_C2'])
    geometry_params.append(params)

unique_params = set(geometry_params)
print(f"总样本数: {len(samples)}")
print(f"唯一几何参数组合数: {len(unique_params)}")

# 显示几个频率点的数据示例
print("\n" + "=" * 80)
print("前3个频率点的数据示例（前3个实部列）:")
print("=" * 80)
for i in range(min(3, len(freq_data))):
    print(f"\n频率 {freq_data[i]:.6f} GHz:")
    for j in range(min(3, len(real_columns))):
        print(f"  实部列 {j+1}: {real_data[i, j]:.6f}")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)