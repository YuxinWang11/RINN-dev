# 绘图规范

## 1. 基本要求

### 1.1 字体设置

```python
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 1.2 图表大小

```python
# 标准单图
plt.figure(figsize=(10, 6))

# 对比图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 多子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```

### 1.3 分辨率

```python
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

---

## 2. S参数绘图

### 2.1 标准S11响应图

```python
def plot_s11_response(freqs, s11_dB, title="S11 Response", save_path=None):
    """
    绘制S11响应曲线
    
    参数:
        freqs: 频率数组 [GHz]
        s11_dB: S11的dB值
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(freqs, s11_dB, 'b-', linewidth=2, label='S11 (dB)')
    
    # 添加参考线
    plt.axhline(y=-10, color='r', linestyle='--', linewidth=1, label='-10 dB')
    plt.axhline(y=-20, color='orange', linestyle='--', linewidth=1, label='-20 dB')
    
    # 标记通带（假设10.7-11.7 GHz）
    plt.axvspan(10.7, 11.7, alpha=0.2, color='green', label='Passband')
    
    # 设置标签
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S11 (dB)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 设置y轴范围（S11为负值，所以反向）
    plt.ylim(-40, 0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 2.2 对比图（预测 vs 真实）

```python
def plot_comparison(freqs, y_true, y_pred, title="Comparison", save_path=None):
    """
    对比真实值和预测值
    
    参数:
        freqs: 频率数组
        y_true: 真实值
        y_pred: 预测值
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Re(S11)
    ax1.plot(freqs, y_true[:, :101], 'b-', linewidth=2, label='True')
    ax1.plot(freqs, y_pred[:, :101], 'r--', linewidth=2, label='Predicted')
    ax1.set_xlabel('Frequency (GHz)', fontsize=12)
    ax1.set_ylabel('Re(S11)', fontsize=12)
    ax1.set_title('Real Part', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Im(S11)
    ax2.plot(freqs, y_true[:, 101:], 'b-', linewidth=2, label='True')
    ax2.plot(freqs, y_pred[:, 101:], 'r--', linewidth=2, label='Predicted')
    ax2.set_xlabel('Frequency (GHz)', fontsize=12)
    ax2.set_ylabel('Im(S11)', fontsize=12)
    ax2.set_title('Imaginary Part', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 2.3 误差图

```python
def plot_error_heatmap(errors, title="Error Distribution", save_path=None):
    """
    绘制误差热图
    
    参数:
        errors: 误差矩阵 [samples, features]
    """
    plt.figure(figsize=(12, 6))
    
    im = plt.imshow(errors, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    plt.colorbar(im, label='Error')
    
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 3. 训练过程绘图

### 3.1 损失曲线

```python
def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    参数:
        history: 字典，包含 'train_loss', 'val_loss', 'epoch'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # 损失曲线
    ax1.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training History', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率曲线（如果有）
    if 'lr' in history:
        ax2.plot(history['epoch'], history['lr'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 4. 几何参数可视化

### 4.1 参数分布图

```python
def plot_param_distribution(X, param_names=None, save_path=None):
    """
    绘制几何参数分布
    
    参数:
        X: 几何参数矩阵 [samples, 8]
        param_names: 参数名称列表
    """
    if param_names is None:
        param_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'l1', 'l2', 'l3']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(8):
        axes[i].hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(param_names[i], fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle('Geometry Parameters Distribution', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 4.2 参数相关性热图

```python
def plot_correlation_matrix(X, param_names=None, save_path=None):
    """
    绘制参数相关性矩阵
    """
    import seaborn as sns
    
    if param_names is None:
        param_names = ['a1', 'a2', 'a3', 'a4', 'a5', 'l1', 'l2', 'l3']
    
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                xticklabels=param_names, yticklabels=param_names,
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Parameter Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 5. 性能指标绘图

### 5.1 NMSE分布

```python
def plot_nmse_distribution(nmse_list, title="NMSE Distribution", save_path=None):
    """
    绘制NMSE分布直方图
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(nmse_list, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(nmse_list), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(nmse_list):.4f}')
    plt.axvline(np.median(nmse_list), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {np.median(nmse_list):.4f}')
    
    plt.xlabel('NMSE', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 5.2 散点图（预测 vs 真实）

```python
def plot_scatter(y_true, y_pred, title="Prediction vs True", save_path=None):
    """
    绘制预测值vs真实值散点图
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # 对角线（完美预测线）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 6. 颜色规范

### 6.1 推荐配色

```python
# 主色
colors = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'success': '#2ca02c',      # 绿色
    'danger': '#d62728',       # 红色
    'warning': '#ffbb78',      # 黄色
    'info': '#17becf',         # 青色
    'true': '#1f77b4',         # 真实值 - 蓝
    'pred': '#ff7f0e',         # 预测值 - 橙
}
```

### 6.2 多线图配色

```python
# 使用tab10配色
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# 或使用自定义配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
```

---

## 7. 保存规范

### 7.1 图片格式

- **PNG**: 用于演示、文档
- **PDF**: 用于论文、印刷
- **SVG**: 用于网页、可编辑

### 7.2 命名规范

```
{类型}_{内容}_{日期}.{格式}

示例：
- s11_response_comparison_20240214.png
- training_loss_history_20240214.pdf
- param_distribution_a1_a5_20240214.svg
```

### 7.3 存储位置

```
results/
├── figures/
│   ├── training/          # 训练相关图
│   ├── evaluation/        # 评估相关图
│   ├── comparison/        # 对比图
│   └── analysis/          # 分析图
└── data/                  # 数据文件
```

---

## 8. 代码组织

### 8.1 绘图函数库

建议创建 `utils/plotting.py`：

```python
"""
绘图工具函数库
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def plot_s11_response(...):
    """绘制S11响应"""
    pass

def plot_comparison(...):
    """绘制对比图"""
    pass

# ... 其他函数
```

---

*文档编号: 04*  
*创建日期: 2026-02-14*  
*主题: 绘图规范*
