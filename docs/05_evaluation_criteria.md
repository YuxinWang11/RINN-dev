# 模型效果评估准则

## 1. 效果好的标准

### 1.1 正向预测 (X → Y)

#### 指标阈值

| 指标 | 优秀 | 良好 | 可接受 | 差 |
|------|------|------|--------|-----|
| **NMSE** | < 0.005 | 0.005-0.01 | 0.01-0.05 | > 0.05 |
| **相对误差** | < 1% | 1-5% | 5-10% | > 10% |
| **R² 分数** | > 0.99 | 0.95-0.99 | 0.90-0.95 | < 0.90 |

#### 具体表现

✅ **优秀模型**：
- NMSE < 0.005（误差小于0.5%）
- S11曲线与真实值几乎重合
- 通带内的特征（峰值、谷值）预测准确
- 实部和虚部都预测准确

✅ **良好模型**：
- NMSE 0.005-0.01（误差0.5%-1%）
- S11曲线整体趋势正确
- 通带内可能存在微小偏差
- 满足工程应用需求

✅ **可接受模型**：
- NMSE 0.01-0.05（误差1%-5%）
- S11曲线整体趋势正确，但局部有明显偏差
- 需要人工检查和微调
- 可用于初步设计

### 1.2 反向设计 (Y → X)

#### 核心准则：回推验证

**必须执行**：生成的X必须正向预测，验证Y_pred与Y_target的差异

#### 指标阈值

| 指标 | 优秀 | 良好 | 可接受 | 差 |
|------|------|------|--------|-----|
| **NMSE** | < 0.01 | 0.01-0.05 | 0.05-0.10 | > 0.10 |
| **几何参数误差** | < 2% | 2-5% | 5-10% | > 10% |
| **满足约束** | 100% | >95% | >90% | < 90% |

#### 具体表现

✅ **优秀反向设计**：
- 回推NMSE < 0.01
- 生成的X在有效范围内
- 生成的X对应的Y与目标Y几乎一致
- 多次采样Z能找到高质量解

✅ **良好反向设计**：
- 回推NMSE 0.01-0.05
- 大部分生成的X在有效范围内
- 可能需要轻微调整
- 采样多个Z能找到可接受解

## 2. 效果差的表现

### 2.1 模型未收敛

❌ **训练损失不下降**
- 训练损失始终很高（> 1.0）
- 验证损失震荡或上升
- 学习率可能过大或过小

❌ **过拟合**
- 训练损失很低，但验证损失很高
- 在训练集上表现好，测试集上表现差
- 需要正则化或早停

❌ **欠拟合**
- 训练和验证损失都很高
- 模型容量不足
- 需要增加模型复杂度或训练时间

### 2.2 可逆性问题

❌ **前向-逆向不一致**
```python
z, _, _ = model(x)
x_recon, _ = model.inverse(z)
mse = torch.mean((x_recon - x) ** 2)
```
- 如果 MSE > 1e-5：可逆性有问题
- 可能原因：数值不稳定、实现错误

❌ **雅可比行列式计算错误**
- log_det 出现 NaN 或 Inf
- 矩阵奇异（det = 0）
- 需要检查scale限制和数值稳定性

### 2.3 反向设计失败

❌ **生成的X超出范围**
- 宽度 < 3.0 mm 或 > 7.0 mm
- 长度 < 5.0 mm 或 > 15.0 mm
- 需要添加约束或后处理

❌ **回推验证失败**
- X生成后，预测的Y与目标Y差异巨大
- NMSE > 0.1
- 模型未学习到有效的反向映射

❌ **所有Z都生成差结果**
- 采样100个Z，全部NMSE > 0.05
- 说明模型训练不充分或OOD问题

## 3. 评估检查清单

### 3.1 训练阶段检查

- [ ] 训练损失稳定下降
- [ ] 验证损失与训练损失接近
- [ ] 无明显的过拟合迹象
- [ ] 学习率适当（不过大或过小）
- [ ] 早停机制正常工作

### 3.2 可逆性验证

```python
def check_invertibility(model, x_test, tolerance=1e-5):
    """检查模型可逆性"""
    model.eval()
    with torch.no_grad():
        z, log_det_forward, _ = model(x_test)
        x_recon, log_det_inverse = model.inverse(z)
        
        mse = torch.mean((x_recon - x_test) ** 2).item()
        
        if mse < tolerance:
            print(f"✅ 可逆性验证通过 (MSE: {mse:.2e})")
            return True
        else:
            print(f"❌ 可逆性验证失败 (MSE: {mse:.2e})")
            return False
```

### 3.3 正向预测评估

```python
def evaluate_forward(model, X_test, Y_test, X_mean, X_std, Y_mean, Y_std):
    """评估正向预测性能"""
    model.eval()
    
    with torch.no_grad():
        # 标准化
        X_norm = (X_test - X_mean) / X_std
        Y_norm = (Y_test - Y_mean) / Y_std
        
        # 采样Z进行预测
        Y_preds = []
        for _ in range(10):  # 采样10次
            Z = torch.randn(X_test.shape[0], z_dim).to(X_test.device)
            input_data = torch.cat([Z, X_norm], dim=-1)
            Y_pred_norm, _ = model.inverse(input_data)
            Y_preds.append(Y_pred_norm)
        
        Y_pred_norm = torch.stack(Y_preds).mean(dim=0)
        Y_pred = Y_pred_norm * Y_std + Y_mean
        
        # 计算指标
        mse = torch.mean((Y_pred - Y_test) ** 2).item()
        nmse = mse / torch.mean(Y_test ** 2).item()
        
        print(f"MSE: {mse:.6f}")
        print(f"NMSE: {nmse:.6f}")
        
        if nmse < 0.005:
            print("✅ 优秀")
        elif nmse < 0.01:
            print("✅ 良好")
        elif nmse < 0.05:
            print("⚠️ 可接受")
        else:
            print("❌ 差")
        
        return nmse
```

### 3.4 反向设计评估

```python
def evaluate_reverse(model, Y_target, X_test, X_mean, X_std, Y_mean, Y_std, 
                     num_z_samples=100):
    """评估反向设计性能"""
    model.eval()
    
    Y_normalized = (Y_target - Y_mean) / Y_std
    
    results = []
    with torch.no_grad():
        for _ in range(num_z_samples):
            # 采样Z并生成X
            Z = torch.randn(1, z_dim).to(Y_target.device)
            input_data = torch.cat([Z, Y_normalized], dim=-1)
            X_norm, _ = model.inverse(input_data)
            X = X_norm * X_std + X_mean
            
            # 回推验证
            X_norm_verify = (X - X_mean) / X_std
            Z_verify = torch.randn(1, z_dim).to(X.device)
            input_verify = torch.cat([Z_verify, X_norm_verify], dim=-1)
            Y_pred_norm, _ = model.inverse(input_verify)
            Y_pred = Y_pred_norm * Y_std + Y_mean
            
            # 计算NMSE
            nmse = torch.mean((Y_pred - Y_target) ** 2) / torch.mean(Y_target ** 2)
            
            results.append({
                'X': X.cpu().numpy(),
                'nmse': nmse.item()
            })
    
    # 找到最佳结果
    best_result = min(results, key=lambda x: x['nmse'])
    
    print(f"最佳 NMSE: {best_result['nmse']:.6f}")
    print(f"平均 NMSE: {np.mean([r['nmse'] for r in results]):.6f}")
    
    # 统计高质量解的数量
    good_count = sum(1 for r in results if r['nmse'] < 0.01)
    print(f"高质量解数量 (NMSE<0.01): {good_count}/{num_z_samples}")
    
    return best_result, results
```

## 4. 数据集质量评估

### 4.1 数据覆盖度

检查训练数据是否覆盖目标响应空间：

```python
def check_data_coverage(Y_train, Y_target, k=10):
    """
    检查训练数据对目标响应的覆盖度
    
    返回:
        coverage_score: 覆盖度分数 (0-1)
        similar_samples: 最相似的k个样本
    """
    from scipy.spatial.distance import cdist
    
    # 计算目标与所有训练样本的距离
    distances = cdist(Y_target.reshape(1, -1), Y_train, metric='euclidean')
    
    # 找到k个最近邻
    k_nearest_idx = np.argsort(distances[0])[:k]
    k_nearest_dist = distances[0][k_nearest_idx]
    
    # 计算覆盖度分数（基于平均距离）
    avg_distance = np.mean(k_nearest_dist)
    max_distance = np.max(distances)
    coverage_score = 1 - (avg_distance / max_distance)
    
    print(f"覆盖度分数: {coverage_score:.4f}")
    print(f"k个最近邻平均距离: {avg_distance:.4f}")
    
    if coverage_score > 0.8:
        print("✅ 覆盖度良好")
    elif coverage_score > 0.5:
        print("⚠️ 覆盖度一般，可能需要扩充数据")
    else:
        print("❌ 覆盖度差，强烈建议扩充数据")
    
    return coverage_score, k_nearest_idx
```

### 4.2 多解检测

检查是否存在多个X对应相同Y的情况：

```python
def detect_multimodal_solutions(X, Y, threshold=0.01):
    """
    检测数据集中是否存在多解
    
    参数:
        X: 几何参数 [N, 8]
        Y: 响应 [N, 202]
        threshold: 判定为"相同Y"的阈值
    
    返回:
        multimodal_count: 多解组数
    """
    from scipy.spatial.distance import pdist, squareform
    
    # 计算Y的成对距离
    y_distances = squareform(pdist(Y, metric='euclidean'))
    
    # 找到Y相似但X不同的样本对
    multimodal_groups = []
    
    for i in range(len(Y)):
        similar_y_idx = np.where(y_distances[i] < threshold)[0]
        if len(similar_y_idx) > 1:
            # 检查对应的X是否不同
            x_group = X[similar_y_idx]
            x_variance = np.var(x_group, axis=0).mean()
            
            if x_variance > 0.1:  # X有明显差异
                multimodal_groups.append({
                    'indices': similar_y_idx,
                    'x_variance': x_variance
                })
    
    print(f"发现 {len(multimodal_groups)} 组多解")
    
    if len(multimodal_groups) > 0:
        print("✅ 数据集存在多解，R-INN的潜变量Z设计合理")
    else:
        print("⚠️ 数据集似乎不存在多解，或者阈值设置不当")
    
    return multimodal_groups
```

## 5. 总结

### 5.1 优秀模型的特征

✅ **训练阶段**：
- 损失稳定下降到较低水平（< 0.1）
- 训练和验证损失接近
- 无过拟合迹象

✅ **可逆性**：
- 前向-逆向MSE < 1e-5
- 雅可比行列式计算稳定

✅ **正向预测**：
- NMSE < 0.01
- S11曲线与真实值高度一致

✅ **反向设计**：
- 回推NMSE < 0.05
- 能找到在有效范围内的X
- 多个Z采样能找到高质量解

### 5.2 需要改进的信号

❌ **立即需要关注**：
- 可逆性验证失败
- NMSE > 0.1
- 所有Z采样都失败
- 生成的X严重超出范围

❌ **需要优化**：
- NMSE 0.05-0.1
- 回推验证偶尔失败
- 数据覆盖度低
- 多解检测不到

---

*文档编号: 05*  
*创建日期: 2026-02-14*  
*主题: 效果评估准则*
