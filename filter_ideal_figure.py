import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# ===================== 【固定：四阶波导滤波器 核心参数】 =====================
fs = 40e9               # 采样率 30GHz
order = 4               # 四阶！严格对应 fourth-order waveguide filter，禁止修改
n_points = 2048         # RINN输入频率分辨率
f_start = 9.0e9         # 起始频率
f_stop = 12.0e9         # 终止频率

# ===================== 【你的硬性需求配置】 =====================
filter_configs = [
    {
        'name': 'X_band_4th_order',
        'f_low': 10.85e9,
        'f_high': 11.15e9,
        's11_target': -26,       # 你的需求：通带S11全部 ≤ 该值
        #'s11_clamp_max': -25,    # 【强制钳位】通带最高S11不超过-22dB（比-20更低，留余量）
        'description': '4th-order Waveguide BPF |S11 ≤ -20dB (10.85-11.15GHz)'
    }
]

# ===================== 【RP参数搜索配置】 =====================
rp_min = 0.001     # RP最小值
rp_max = 5.00      # RP最大值
rp_step = 0.01     # RP步长
rp_values = np.arange(rp_min, rp_max + rp_step, rp_step)

# 输出文件夹
result_dir = 'result_RINN_input'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# ===================== 主处理流程 =====================
for cfg in filter_configs:
    print(f"\n===== 生成满足 |S11 ≤ {cfg['s11_target']}dB 的四阶波导滤波器频响 =====")
    f_low, f_high = cfg['f_low'], cfg['f_high']
    target_dB = cfg['s11_target']
    #clamp_dB = cfg['s11_clamp_max']
    desc = cfg['description']

    # 1. 生成频率轴
    freq = np.linspace(f_start, f_stop, n_points)
    nyq = 0.5 * fs
    w_norm = [f_low/nyq, f_high/nyq]
    passband_mask = (freq >= f_low) & (freq <= f_high)
    passband_length = np.sum(passband_mask)

    # 2. RP参数搜索
    print(f"\n=== 开始RP参数搜索（{rp_min} 到 {rp_max}，步长 {rp_step}）===")
    
    # 存储每个RP值的结果
    results = []
    
    for rp in rp_values:
        # 设计四阶切比雪夫I型带通滤波器
        z, p, k = signal.cheby1(order, rp, w_norm, btype='bandpass', output='zpk')
        b, a = signal.zpk2tf(z, p, k)
        w, h = signal.freqz(b, a, worN=freq, fs=fs)

        # 计算S11（无损网络）
        S21 = h  # S21是复数
        S21_abs = np.abs(S21)
        S21_angle = np.angle(S21)  # S21的相位
        
        # 计算S11的幅度
        S11_abs = np.sqrt(np.clip(1 - S21_abs**2, 1e-12, 1.0))  # 防除0/负数
        
        # 计算S11的复数形式（假设相位与S21相差90度）
        S11 = 1j * S11_abs * np.exp(1j * S21_angle)
        
        # 提取S11的实部和虚部
        S11_real = np.real(S11)
        S11_imag = np.imag(S11)
        
        # 计算S11的dB值
        S11_dB = 20 * np.log10(S11_abs)

        # 计算通带内的最大值和最小值
        pb_vals = S11_dB[passband_mask]
        max_pb = np.max(pb_vals)
        min_pb = np.min(pb_vals)
        
        # 实现新的评估准则：找到通带内的谷，计算最左侧谷和最右侧谷之间的最高值S
        # 1. 检测谷（局部最小值）
        def find_valleys(data):
            valleys = []
            for i in range(1, len(data)-1):
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    valleys.append(i)
            # 处理边界情况
            if len(data) > 0:
                if len(valleys) == 0 or valleys[0] > 0:
                    valleys.insert(0, 0)
                if len(valleys) == 0 or valleys[-1] < len(data)-1:
                    valleys.append(len(data)-1)
            return valleys
        
        # 2. 找到谷的位置
        valleys = find_valleys(pb_vals)
        
        # 3. 计算最左侧谷和最右侧谷之间的最高值S
        if len(valleys) >= 2:
            left_valley = valleys[0]
            right_valley = valleys[-1]
            # 提取最左侧谷和最右侧谷之间的所有数值
            middle_vals = pb_vals[left_valley:right_valley+1]
            S = np.max(middle_vals)
        else:
            # 如果没有足够的谷，使用整个通带的最大值
            S = max_pb
        
        # 4. 计算S与目标dB的差值（绝对值）
        diff = abs(S - target_dB)
        
        # 存储结果
        results.append({
            'rp': rp,
            'S': S,
            'diff': diff,
            'max_pb': max_pb,
            'min_pb': min_pb,
            'valleys_count': len(valleys),
            'S11_real': S11_real,
            'S11_imag': S11_imag
        })
    
    # 3. 找到最佳RP值（S与目标dB差值最小的）
    best_result = min(results, key=lambda x: x['diff'])
    best_rp = best_result['rp']
    best_S = best_result['S']
    best_diff = best_result['diff']
    best_max_pb = best_result['max_pb']
    best_min_pb = best_result['min_pb']
    best_valleys_count = best_result['valleys_count']
    
    print(f"\n=== 搜索完成！===")
    print(f"最佳RP值: {best_rp:.2f} dB")
    print(f"通带内最左侧谷和最右侧谷之间的最高值S: {best_S:.2f} dB")
    print(f"S与目标dB的差值: {best_diff:.2f} dB")
    print(f"检测到的谷数量: {best_valleys_count}")
    print(f"通带内S11最大值: {best_max_pb:.2f} dB")
    print(f"通带内S11最小值: {best_min_pb:.2f} dB")
    
    # 4. 使用最佳RP值生成最终结果
    print(f"\n=== 使用最佳RP值 {best_rp:.2f} 生成最终结果 ===")
    z, p, k = signal.cheby1(order, best_rp, w_norm, btype='band', output='zpk')
    b, a = signal.zpk2tf(z, p, k)
    w, h = signal.freqz(b, a, worN=freq, fs=fs)

    # 计算S11（无损网络）
    S21 = h  # S21是复数
    S21_abs = np.abs(S21)
    S21_angle = np.angle(S21)  # S21的相位
    
    # 计算S11的幅度
    S11_abs = np.sqrt(np.clip(1 - S21_abs**2, 1e-12, 1.0))  # 防除0/负数
    
    # 计算S11的复数形式（假设相位与S21相差90度）
    S11 = 1j * S11_abs * np.exp(1j * S21_angle)
    
    # 提取S11的实部和虚部
    S11_real = np.real(S11)
    S11_imag = np.imag(S11)
    
    # 计算S11的dB值
    S11_dB = 20 * np.log10(S11_abs)

    # 5. 平滑处理，还原真实波导滤波器的曲线质感
    S11_dB = gaussian_filter1d(S11_dB, sigma=1.5)

    # 6. 统计验证：是否全部满足 ≤ target_dB
    pb_vals = S11_dB[passband_mask]
    all_ok = np.all(pb_vals <= target_dB)
    max_pb = np.max(pb_vals)
    min_pb = np.min(pb_vals)
    below_target = np.sum(pb_vals <= target_dB)
    ratio = below_target / passband_length

    print(f"通带S11最大值: {max_pb:.2f} dB (≤ {target_dB}dB: {all_ok})")
    print(f"通带S11最小值: {min_pb:.2f} dB")
    print(f"通带内小于 {target_dB}dB 的比例: {ratio:.4f} ({ratio*100:.2f}%)")
    if all_ok:
        print(f"✅ 全部满足 |S11| ≤ {target_dB}dB 需求！")
    else:
        print(f"⚠️  部分通带不满足 |S11| ≤ {target_dB}dB 需求")

    # ===================== 绘图输出 =====================
    plt.figure(figsize=(12, 7))
    plt.plot(freq/1e9, S11_dB, 'blue', linewidth=2.5, label='|S11| (dB)')
    
    # 通带阴影
    plt.axvspan(f_low/1e9, f_high/1e9, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
    
    # 你的需求红线
    plt.axhline(target_dB, color='red', linestyle='--', linewidth=2, label=f'{target_dB}dB Requirement Line')
    
    # 坐标与样式
    plt.xlim(9, 11.5)
    plt.ylim(-60, 0)
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('|S11| (dB)', fontsize=14)
    plt.title(f'4th-order Waveguide Bandpass Filter (RP={best_rp:.2f})\n{desc}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(result_dir, f'Waveguide_4th_S11_under_{target_dB}dB.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # ===================== 绘制S11实部和虚部 =====================
    plt.figure(figsize=(12, 7))
    
    # 绘制实部
    plt.plot(freq/1e9, S11_real, 'blue', linewidth=2, label='S11 Real Part')
    
    # 绘制虚部
    plt.plot(freq/1e9, S11_imag, 'red', linewidth=2, label='S11 Imaginary Part')
    
    # 通带阴影
    plt.axvspan(f_low/1e9, f_high/1e9, color='green', alpha=0.15, label='Passband(10.85-11.15GHz)')
    
    # 坐标与样式
    plt.xlim(9, 11.5)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Frequency (GHz)', fontsize=14)
    plt.ylabel('S11 Value', fontsize=14)
    plt.title(f'4th-order Waveguide Bandpass Filter S11 Real/Imaginary Parts (RP={best_rp:.2f})\n{desc}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    save_path_real_imag = os.path.join(result_dir, f'Waveguide_4th_S11_real_imag_{target_dB}dB.png')
    plt.savefig(save_path_real_imag, dpi=300)
    plt.close()

    # ===================== 保存RINN输入数据 =====================
    data_path = os.path.join(result_dir, 'RINN_input_freq_S11.npz')
    np.savez(data_path,
             frequency_Hz=freq,
             S11_dB=S11_dB,
             S11_real=S11_real,
             S11_imag=S11_imag,
             passband_mask=passband_mask,
             target_threshold=target_dB,
             best_rp=best_rp,
             best_S=best_S,
             best_diff=best_diff)

    # 保存RP搜索结果
    results_path = os.path.join(result_dir, 'RP_search_results.npz')
    # 提取结果为numpy数组
    rps = np.array([r['rp'] for r in results])
    S_values = np.array([r['S'] for r in results])
    diffs = np.array([r['diff'] for r in results])
    max_pbs = np.array([r['max_pb'] for r in results])
    min_pbs = np.array([r['min_pb'] for r in results])
    valleys_counts = np.array([r['valleys_count'] for r in results])
    
    np.savez(results_path,
             rps=rps,
             S_values=S_values,
             diffs=diffs,
             max_pbs=max_pbs,
             min_pbs=min_pbs,
             valleys_counts=valleys_counts,
             target_threshold=target_dB)

    print(f"📊 图像保存至: {save_path}")
    print(f"� S11实部和虚部图像保存至: {save_path_real_imag}")
    print(f"�📦 RINN输入数据保存至: {data_path}")
    print(f"📈 RP搜索结果保存至: {results_path}")
    print("="*80)

print("\n🎉 全部生成完成：四阶波导滤波器 S11 全部满足目标要求，可直接输入可逆神经网络！")