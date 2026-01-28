#!/usr/bin/env python3
"""
RINN模型实验运行脚本

本脚本用于接收JSON格式的配置参数，并调用trains11RINN.py的功能来运行实验。
用户可以通过修改config字典中的参数来快速调整实验设置。
"""

import os
import json
import subprocess
import sys
from datetime import datetime

# 默认配置参数
config = {
    "model_config": {
        "hidden_dim": 60,
        "num_blocks": 5,
        "num_stages": 3,
        "num_cycles_per_stage": 3,
        "ratio_toZ_after_flowstage": 0.3,
        "ratio_x1_x2_inAffine": 0.04716981132075472
    },
    "training_params": {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 1e-05,
        "clip_value": 0.5,
        "num_epochs": 100,
        "loss_weights": {
            "weight_y": 1.9,
            "weight_x": 0.5,
            "weight_z": 0.3
        }
    },
    "data_params": {
        "normalization_method": "robust"
    }
}


def run_experiment(config):
    """
    运行RINN模型实验
    
    Args:
        config: 包含模型配置和训练参数的字典
    """
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"rinn_experiment_{timestamp}"
    experiment_dir = os.path.join('model_checkpoints_rinn', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 开始RINN模型实验 ===")
    print(f"实验名称: {experiment_name}")
    print(f"配置文件: {config_path}")
    print(f"\n模型配置:")
    print(json.dumps(config['model_config'], indent=4))
    print(f"\n训练参数:")
    print(json.dumps(config['training_params'], indent=4))
    
    # 运行trains11RINN.py脚本
    print(f"\n正在执行trains11RINN.py...")
    
    # 将配置文件传递给trains11RINN.py
    result = subprocess.run(
        [sys.executable, 'trains11RINN.py', '--config', config_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # 输出执行结果
    print(f"\n=== 实验执行结果 ===")
    print(f"返回码: {result.returncode}")
    
    if result.stdout:
        print(f"\n标准输出:")
        print(result.stdout)
    
    if result.stderr:
        print(f"\n标准错误:")
        print(result.stderr)
    
    # 将执行结果保存到实验目录
    output_path = os.path.join(experiment_dir, 'execution_output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"返回码: {result.returncode}\n\n")
        f.write("标准输出:\n")
        f.write(result.stdout)
        f.write("\n标准错误:\n")
        f.write(result.stderr)
    
    print(f"\n执行结果已保存到: {output_path}")
    print(f"\n=== 实验完成 ===")


if __name__ == "__main__":
    # 显示当前配置
    print("=== RINN模型实验配置 ===")
    print(json.dumps(config, indent=4))
    
    # 直接运行实验
    run_experiment(config)