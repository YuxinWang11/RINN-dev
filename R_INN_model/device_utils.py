import torch


def get_device():
    """
    获取可用的计算设备，优先使用GPU
    
    Returns:
        torch.device: 可用的计算设备
    """
    # 检查是否有可用的CUDA设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        # 获取当前使用的GPU索引
        current_device = torch.cuda.current_device()
        # 获取GPU型号
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"检测到GPU设备:")
        print(f"  GPU总数: {gpu_count}")
        print(f"  当前使用GPU索引: {current_device}")
        print(f"  GPU型号: {gpu_name}")
        print(f"  CUDA版本: {torch.version.cuda}")
        
        # 显示GPU内存信息
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1024**3
        free_memory = reserved_memory - allocated_memory
        
        print(f"  GPU内存: 总计 {total_memory:.2f} GB")
        print(f"  已分配: {allocated_memory:.2f} GB")
        print(f"  可用: {free_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        print("未检测到GPU设备，使用CPU进行计算")
    
    return device


def test_device_computation(device):
    """
    测试在指定设备上的简单计算
    
    Args:
        device: 要测试的计算设备
    """
    print(f"\n在设备 {device} 上测试计算...")
    
    # 创建一个随机张量并移至指定设备
    x = torch.randn(1000, 1000, device=device)
    
    # 执行简单计算
    y = torch.matmul(x, x.t())
    z = y.mean()
    
    print(f"计算完成! 结果: {z.item():.4f}")


if __name__ == "__main__":
    print("===== 设备检测工具 =====")
    device = get_device()
    test_device_computation(device)
    print("\n设备检测完成!")