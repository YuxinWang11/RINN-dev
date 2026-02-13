# R-INN 工作流程说明书

## 概述

本文档描述使用R-INN模型进行**正向预测**和**反向设计**的完整工作流程。

---

## 一、正向预测：X → Y

### 1.1 流程图

```
几何参数 X
    ↓
标准化 (使用训练集的均值和标准差)
    ↓
随机采样 Z ~ N(0,1)
    ↓
拼接 [X, Z]
    ↓
输入 R-INN
    ↓
逆向变换 model.inverse([Z, ?])
    ↓
输出 Y (电磁响应)
    ↓
反标准化 (恢复原始尺度)
    ↓
得到预测的S参数
```

### 1.2 详细步骤

#### 步骤1：准备几何参数 X

```python
X = [a₁, a₂, a₃, a₄, a₅, l₁, l₂, l₃]  # 8维向量
```

确保参数在有效范围内：
- 宽度：3.0-7.0 mm
- 长度：5.0-15.0 mm

#### 步骤2：数据标准化

```python
X_normalized = (X - X_mean) / X_std
```

使用训练集计算得到的均值和标准差。

#### 步骤3：采样潜变量 Z

```python
Z = torch.randn(batch_size, z_dim)
```

`z_dim` 取决于模型架构。

#### 步骤4：模型推理

```python
# 拼接输入
input_data = torch.cat([Z, X_normalized], dim=-1)

# 逆向变换
Y_normalized, _ = model.inverse(input_data)

# 反标准化
Y = Y_normalized * Y_std + Y_mean
```

#### 步骤5：解析输出

```python
# 当前方案 (202维)
Re_S11 = Y[:, :101]  # 前101维：实部
Im_S11 = Y[:, 101:]  # 后101维：虚部

# 计算幅度和dB
mag_S11 = np.sqrt(Re_S11**2 + Im_S11**2)
dB_S11 = 20 * np.log10(mag_S11)
```

### 1.3 代码示例

```python
def predict_forward(model, X, Y_mean, Y_std, X_mean, X_std, num_samples=10):
    """
    正向预测：X → Y
    
    参数:
        model: 训练好的R-INN模型
        X: 几何参数 [batch_size, 8]
        Y_mean, Y_std: Y的统计信息
        X_mean, X_std: X的统计信息
        num_samples: 采样Z的次数
    
    返回:
        Y_pred: 预测的Y [num_samples, batch_size, 202]
    """
    model.eval()
    X_normalized = (X - X_mean) / X_std
    
    Y_samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            # 随机采样Z
            Z = torch.randn(X.shape[0], z_dim).to(X.device)
            
            # 拼接 [Z, X]
            input_data = torch.cat([Z, X_normalized], dim=-1)
            
            # 逆向变换
            Y_norm, _ = model.inverse(input_data)
            
            # 反标准化
            Y = Y_norm * Y_std + Y_mean
            Y_samples.append(Y)
    
    return torch.stack(Y_samples)
```

---

## 二、反向设计：Y → X

### 2.1 流程图

```
目标响应 Y_target
    ↓
标准化
    ↓
随机采样 Z ~ N(0,1)
    ↓
拼接 [Z, Y_target]
    ↓
输入 R-INN
    ↓
逆向变换 model.inverse([Z, Y_target])
    ↓
输出 X_candidate (几何参数候选)
    ↓
反标准化
    ↓
约束到有效范围 (clip/tanh)
    ↓
得到候选几何参数
    ↓
【可选】正向验证：X_candidate → Y_pred
    ↓
比较 Y_pred vs Y_target
    ↓
评估是否接受此候选
```

### 2.2 详细步骤

#### 步骤1：准备目标响应 Y

```python
Y_target = [Re_S11_1, ..., Re_S11_101, Im_S11_1, ..., Im_S11_101]
```

#### 步骤2：标准化

```python
Y_normalized = (Y_target - Y_mean) / Y_std
```

#### 步骤3：采样Z并生成候选X

```python
# 采样多个Z，生成多个候选
X_candidates = []
for _ in range(num_z_samples):
    Z = torch.randn(1, z_dim)
    input_data = torch.cat([Z, Y_normalized], dim=-1)
    X_norm, _ = model.inverse(input_data)
    X = X_norm * X_std + X_mean
    X_candidates.append(X)
```

#### 步骤4：约束到有效范围

```python
# 方法1：硬裁剪
X_clipped = torch.clamp(X, X_min, X_max)

# 方法2：tanh约束
X_constrained = X_min + (X_max - X_min) * (torch.tanh(X) + 1) / 2
```

#### 步骤5：验证（关键！）

```python
# 用生成的X重新预测Y
Y_pred = predict_forward(model, X_constrained, ...)

# 计算误差
error = torch.mean((Y_pred - Y_target) ** 2)

# 评估NMSE
nmse = error / torch.mean(Y_target ** 2)
```

### 2.3 代码示例

```python
def design_reverse(model, Y_target, X_mean, X_std, Y_mean, Y_std, 
                   num_samples=100, X_min=None, X_max=None):
    """
    反向设计：Y → X
    
    参数:
        model: 训练好的R-INN模型
        Y_target: 目标响应 [1, 202]
        X_mean, X_std, Y_mean, Y_std: 统计信息
        num_samples: 采样Z的次数
        X_min, X_max: X的范围约束
    
    返回:
        best_X: 最佳几何参数 [1, 8]
        best_nmse: 对应的NMSE
        all_results: 所有候选的结果
    """
    model.eval()
    Y_normalized = (Y_target - Y_mean) / Y_std
    
    results = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # 采样Z
            Z = torch.randn(1, z_dim).to(Y_target.device)
            
            # 拼接 [Z, Y]
            input_data = torch.cat([Z, Y_normalized], dim=-1)
            
            # 逆向变换得到X
            X_norm, _ = model.inverse(input_data)
            X = X_norm * X_std + X_mean
            
            # 约束到有效范围
            if X_min is not None and X_max is not None:
                X = torch.clamp(X, X_min, X_max)
            
            # 【关键】正向验证
            Y_pred = predict_forward(model, X, Y_mean, Y_std, X_mean, X_std, num_samples=1)
            
            # 计算NMSE
            nmse = torch.mean((Y_pred - Y_target) ** 2) / torch.mean(Y_target ** 2)
            
            results.append({
                'X': X.cpu().numpy(),
                'nmse': nmse.item(),
                'Z': Z.cpu().numpy()
            })
    
    # 按NMSE排序，返回最佳结果
    results.sort(key=lambda x: x['nmse'])
    best_result = results[0]
    
    return best_result['X'], best_result['nmse'], results
```

---

## 三、贝叶斯优化流程

### 3.1 应用场景

当R-INN生成的X质量不够好时，使用贝叶斯优化微调。

### 3.2 流程

```python
from bayes_opt import BayesianOptimization

def objective_function(a1, a2, a3, a4, a5, l1, l2, l3):
    """目标函数：最小化预测Y与目标Y的差异"""
    X = torch.tensor([[a1, a2, a3, a4, a5, l1, l2, l3]])
    Y_pred = predict_forward(model, X, ...)
    error = -torch.mean((Y_pred - Y_target) ** 2)  # 负值，因为bayes_opt求最大值
    return error.item()

# 定义搜索空间
pbounds = {
    'a1': (3.0, 7.0), 'a2': (3.0, 7.0), 'a3': (3.0, 7.0),
    'a4': (3.0, 7.0), 'a5': (3.0, 7.0),
    'l1': (5.0, 15.0), 'l2': (5.0, 15.0), 'l3': (5.0, 15.0)
}

# 初始化优化器
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)

# 执行优化
optimizer.maximize(init_points=10, n_iter=50)

# 最佳结果
best_params = optimizer.max['params']
```

---

## 四、完整示例：从目标到设计

```python
# 1. 加载模型和数据
checkpoint = torch.load('model_checkpoints_rinn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
X_mean, X_std = checkpoint['X_mean'], checkpoint['X_std']
Y_mean, Y_std = checkpoint['Y_mean'], checkpoint['Y_std']

# 2. 定义目标响应（如理想滤波器）
Y_target = generate_ideal_filter(target_dB=-26)  # 自定义函数

# 3. 反向设计
best_X, best_nmse, all_results = design_reverse(
    model, Y_target, X_mean, X_std, Y_mean, Y_std,
    num_samples=1000,  # 采样1000个Z
    X_min=X_min, X_max=X_max
)

print(f"最佳NMSE: {best_nmse:.6f}")
print(f"最佳几何参数: {best_X}")

# 4. 【可选】贝叶斯优化微调
if best_nmse > threshold:
    refined_X = bayesian_optimize(model, Y_target, best_X, pbounds)
    print(f"优化后参数: {refined_X}")

# 5. 最终验证
Y_final = predict_forward(model, best_X, ...)
plot_comparison(Y_target, Y_final)  # 可视化对比
```

---

## 五、关键要点

### 5.1 必须做的事情

✅ **始终使用标准化**：输入模型前标准化，输出后反标准化  
✅ **采样多个Z**：一个Y对应多个X，采样多个Z找最佳  
✅ **正向验证**：生成的X必须正向验证，计算NMSE  
✅ **约束范围**：确保生成的X在有效范围内  
✅ **检查可逆性**：定期验证模型的可逆性

### 5.2 不要做的事情

❌ 不要直接使用未标准化的数据  
❌ 不要只采样一个Z就停止  
❌ 不要忽视正向验证的NMSE  
❌ 不要生成超出范围的X而不处理  

---

## 六、故障排除

### Q1: 生成的X对应的Y与目标Y差异大？

**原因**：
- 训练数据未覆盖目标Y（OOD问题）
- Z采样不够多
- 模型训练不充分

**解决**：
- 增加Z采样数量
- 使用贝叶斯优化微调
- 扩充训练数据

### Q2: 生成的X超出有效范围？

**原因**：
- 模型未学习到范围约束
- 数据分布不均匀

**解决**：
- 使用tanh/sigmoid限制
- 后处理裁剪
- 在训练时添加约束损失

### Q3: NMSE很高？

**检查清单**：
- [ ] 标准化/反标准化是否正确？
- [ ] 模型是否训练充分？
- [ ] 是否采样了足够多的Z？
- [ ] 目标Y是否在训练数据分布内？

---

*文档编号: 03*  
*创建日期: 2026-02-14*  
*主题: 工作流程*
