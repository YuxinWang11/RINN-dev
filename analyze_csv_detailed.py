"""
详细分析 S Parameter Plot 7.csv 文件结构
理解实部和虚部的对应关系
"""
import csv
import numpy as np
import re

# 读取CSV文件
data_path = 'data/S Parameter Plot 7.csv'

# 读取表头
with open(data_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)

print("=" * 80)
print("详细分析表头结构")
print("=" * 80)

# 显示前20列的表头
print(f"\n前20列表头:")
for i, col in enumerate(header[:20]):
    print(f"列 {i}: {col[:80]}...")

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

# 分析前10个数据列
print("\n" + "=" * 80)
print("前10个数据列的详细信息:")
print("=" * 80)
for i in range(1, min(11, len(header))):
    col = header[i]
    params = extract_geometry_params(col)
    if params:
        print(f"\n列 {i}:")
        print(f"  完整名称: {col[:100]}")
        print(f"  H1: {params['H1']:.5f} mm")
        print(f"  H2: {params['H2']:.5f} mm")
        print(f"  H3: {params['H3']:.5f} mm")
        print(f"  H_C1: {params['H_C1']:.5f} mm")
        print(f"  H_C2: {params['H_C2']:.5f} mm")
        print(f"  类型: {params['type']}")

# 按几何参数分组
print("\n" + "=" * 80)
print("按几何参数分组分析:")
print("=" * 80)

geometry_dict = {}
for i, col in enumerate(header[1:], start=1):
    params = extract_geometry_params(col)
    if params:
        # 创建几何参数的键
        geo_key = (params['H1'], params['H2'], params['H3'], params['H_C1'], params['H_C2'])
        
        if geo_key not in geometry_dict:
            geometry_dict[geo_key] = {'real': None, 'imag': None}
        
        if params['type'] == 'real':
            geometry_dict[geo_key]['real'] = i
        else:
            geometry_dict[geo_key]['imag'] = i

print(f"唯一几何参数组合数: {len(geometry_dict)}")

# 检查有多少组合同时有实部和虚部
both_count = sum(1 for v in geometry_dict.values() if v['real'] is not None and v['imag'] is not None)
real_only_count = sum(1 for v in geometry_dict.values() if v['real'] is not None and v['imag'] is None)
imag_only_count = sum(1 for v in geometry_dict.values() if v['real'] is None and v['imag'] is not None)

print(f"同时有实部和虚部的组合: {both_count}")
print(f"只有实部的组合: {real_only_count}")
print(f"只有虚部的组合: {imag_only_count}")

# 显示几个同时有实部和虚部的组合
print("\n" + "=" * 80)
print("前5个同时有实部和虚部的几何参数组合:")
print("=" * 80)
count = 0
for geo_key, cols in geometry_dict.items():
    if cols['real'] is not None and cols['imag'] is not None:
        count += 1
        if count <= 5:
            print(f"\n组合 {count}:")
            print(f"  H1: {geo_key[0]:.5f} mm")
            print(f"  H2: {geo_key[1]:.5f} mm")
            print(f"  H3: {geo_key[2]:.5f} mm")
            print(f"  H_C1: {geo_key[3]:.5f} mm")
            print(f"  H_C2: {geo_key[4]:.5f} mm")
            print(f"  实部列索引: {cols['real']}")
            print(f"  虚部列索引: {cols['imag']}")

# 读取数据并验证
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
freq_data = data[:, 0]

print("\n" + "=" * 80)
print("数据验证:")
print("=" * 80)
print(f"频率点数: {len(freq_data)}")
print(f"频率范围: {freq_data[0]:.6f} - {freq_data[-1]:.6f} GHz")

# 验证第一个同时有实部和虚部的组合
for geo_key, cols in geometry_dict.items():
    if cols['real'] is not None and cols['imag'] is not None:
        print(f"\n验证几何参数组合:")
        print(f"  H1: {geo_key[0]:.5f} mm")
        print(f"  H2: {geo_key[1]:.5f} mm")
        print(f"  H3: {geo_key[2]:.5f} mm")
        print(f"  H_C1: {geo_key[3]:.5f} mm")
        print(f"  H_C2: {geo_key[4]:.5f} mm")
        
        real_col_data = data[:, cols['real']]
        imag_col_data = data[:, cols['imag']]
        
        print(f"\n实部数据（前5个频率点）:")
        for i in range(min(5, len(freq_data))):
            print(f"  {freq_data[i]:.6f} GHz: {real_col_data[i]:.6f}")
        
        print(f"\n虚部数据（前5个频率点）:")
        for i in range(min(5, len(freq_data))):
            print(f"  {freq_data[i]:.6f} GHz: {imag_col_data[i]:.6f}")
        
        break

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)