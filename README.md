
# R-INN 论文复现项目 README

## 最新进展

该分支的最新进展在trains11RINN_SParameter.py中,该文件有详细的开头注释.

实验结果在model_checkpoints_rinn/rinn_sparameter_20260128_000039中.

该分支的主旨思想: 曾经为5维几何参数,预测101维的性能图,横轴为频率f.经过老师的意见,我们决定改为202维的性能图,即不使用幅值,而是实部和虚部,并且将原先的横轴f,拼接在5维几何参数后侧,最后的模型形状即为左侧5(几何参数)+1(频率f)+2(零填充),右侧2(实数和虚数)+6(latent变量Z).

### 最佳模型性能
- **最佳Epoch**: 72
- **最佳验证集损失**: 0.008595
- **验证集Y损失**: 0.001208
- **验证集Y NMSE**: 0.001251
- **验证集X损失**: 0.000092
- **验证集Z损失**: 0.022458
​
## 项目概述​
本仓库用于复现论文《R-INN: An Efficient Reversible Design Model for Microwave Circuit Design》中的可逆神经网络（Real NVP-based Invertible Neural Network）模型。该模型将可逆神经网络应用于微波电路设计，通过学习电路参数与电磁响应之间的映射关系，实现高效的电路设计与优化。​
复现目标包括：​
实现论文提出的 R-INN 网络结构​
复现关键实验结果（如 NMSE 误差、S 参数拟合曲线）​
建立可复用的微波电路设计深度学习工作流


# 环境配置&项目页面


<sup> https://www.notion.so/276a5242543380d38f58f1346d75a53d?source=copy_link </sup>

<sup> https://www.notion.so/R-INN-271a52425433805790a7c71d2da181dd </sup>
