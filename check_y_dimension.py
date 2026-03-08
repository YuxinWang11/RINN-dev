import csv
import numpy as np
import re

def extract_geometry_params(col_name):
    col_name = col_name.replace('\n', '').replace('"', '')
    h1_match = re.search(r"H1='([\d.]+)mm'", col_name)
    h2_match = re.search(r"H2='([\d.]+)mm'", col_name)
    h3_match = re.search(r"H3='([\d.]+)mm'", col_name)
    hc1_match = re.search(r"H_C1='([\d.]+)mm'", col_name)
    hc2_match = re.search(r"H_C2='([\d.]+)mm'", col_name)
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
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    geometry_dict = {}
    for i, col in enumerate(header[1:]):
        params = extract_geometry_params(col)
        if params:
            geo_key = tuple(params['params'])
            if geo_key not in geometry_dict:
                geometry_dict[geo_key] = {'real': None, 'imag': None}
            if params['type'] == 'real':
                geometry_dict[geo_key]['real'] = i+1
            else:
                geometry_dict[geo_key]['imag'] = i+1
    
    valid_samples = []
    real_columns = []
    imag_columns = []
    
    for geo_key, cols in geometry_dict.items():
        if cols['real'] is not None and cols['imag'] is not None:
            valid_samples.append(list(geo_key))
            real_columns.append(cols['real'])
            imag_columns.append(cols['imag'])
    
    x_features = np.array(valid_samples, dtype=np.float32)
    
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    freq_data = data[:, 0]
    
    real_data = data[:, real_columns]
    imag_data = data[:, imag_columns]
    
    real_data = real_data.T
    imag_data = imag_data.T
    
    y_data = np.concatenate((real_data, imag_data), axis=1)
    return x_features, y_data, freq_data

train_x, train_y, freq_data = load_data_from_csv('data/S Parameter Plot300.csv')
print('=== 实际数据维度 ===')
print(f'X维度: {train_x.shape[1]}')
print(f'Y维度: {train_y.shape[1]}')
print(f'频率点数: {len(freq_data)}')
print(f'Y数据形状: {train_y.shape}')
print(f'  实部维度: {train_y.shape[1] // 2}')
print(f'  虚部维度: {train_y.shape[1] // 2}')
print(f'  总维度: {train_y.shape[1]}')
print(f'  正确说法: {train_y.shape[1] // 2}维实部 + {train_y.shape[1] // 2}维虚部 = {train_y.shape[1]}维')
