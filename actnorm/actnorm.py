import torch


__all__ = ["ActNorm1d", "ActNorm2d", "ActNorm3d"]


class ActNorm(torch.jit.ScriptModule):
    def __init__(self, num_features: int): #num_features：特征数，通道数。
        super().__init__()
        #是否可以改成初始值为随机小数值，防止梯度消失————————————————————————？？？？————————————————————————————————
        self.scale = torch.nn.Parameter(torch.zeros(num_features))#学习参数，初始值全 0，最终是 “1 / 输入标准差，学的是谁
        self.bias = torch.nn.Parameter(torch.zeros(num_features)) #学习参数，初始值全 0，最终是 “- 均值 / 标准差，学的是谁的额？
        self.register_buffer("_initialized", torch.tensor(False)) #标记是否完成初始化

    def reset_(self):
        self._initialized = torch.tensor(False)
        return self

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        正向变换: y = scale * x + bias
        
        Args:
            x: 输入张量
            
        Returns:
            变换后的张量
        """
        self._check_input_dim(x) #检查维度
        if x.dim() > 2:
            x = x.transpose(1, -1)#维度转换 nchw->nhwc
        if not self._initialized:
            # 计算均值和标准差，避免batch size为1时的问题
            flat_x = x.detach().reshape(-1, x.shape[-1])
            mean_x = flat_x.mean(0)
            std_x = flat_x.std(0, unbiased=False)
            
            # 避免标准差为0的情况
            std_x = torch.clamp(std_x, min=1e-6)
            
            self.scale.data = 1.0 / std_x
            self.bias.data = -self.scale * mean_x
            self._initialized = torch.tensor(True)
        x = self.scale * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        逆向变换: x = (y - bias) / scale
        
        Args:
            y: 输出张量（需要逆变换的张量）
            
        Returns:
            逆变换后的张量
            
        Raises:
            RuntimeError: 如果ActNorm层尚未初始化
        """
        if not self._initialized:
            raise RuntimeError("ActNorm layer must be initialized before calling inverse(). "
                             "Please run forward() with some data first.")
        
        self._check_input_dim(y)
        
        # 处理多维张量的维度转换
        if y.dim() > 2:
            y = y.transpose(1, -1)
        
        # 执行逆向仿射变换: x = (y - bias) / scale
        x = (y - self.bias) / self.scale
        
        # 恢复原始维度顺序
        if x.dim() > 2:
            x = x.transpose(1, -1)
            
        return x

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算雅可比行列式的对数
        
        对于仿射变换 y = scale * x + bias，
        雅可比矩阵是对角矩阵，对角元素为 scale
        因此 log|det(J)| = sum(log|scale|)
        
        Args:
            x: 输入张量
            
        Returns:
            雅可比行列式对数的标量张量
        """
        if not self._initialized:
            raise RuntimeError("ActNorm layer must be initialized before calling log_det_jacobian(). "
                             "Please run forward() with some data first.")
        
        self._check_input_dim(x)
        
        # 计算批次大小和空间维度
        batch_size = x.shape[0]
        
        if x.dim() == 2:
            # 1D case: (N, C)
            spatial_dims = 1
        elif x.dim() == 3:
            # 1D sequence case: (N, C, L)
            spatial_dims = x.shape[2]
        elif x.dim() == 4:
            # 2D case: (N, C, H, W)
            spatial_dims = x.shape[2] * x.shape[3]
        elif x.dim() == 5:
            # 3D case: (N, C, D, H, W)
            spatial_dims = x.shape[2] * x.shape[3] * x.shape[4]
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # log|det(J)| = spatial_dims * sum(log|scale|)
        log_det = spatial_dims * torch.sum(torch.log(torch.abs(self.scale)))
        
        # 返回批次大小的张量，每个样本的log_det相同
        return log_det.expand(batch_size)


class ActNorm1d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class ActNorm2d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class ActNorm3d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
