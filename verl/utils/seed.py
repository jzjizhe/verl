import random
import numpy as np
import torch
import os

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    设置所有随机数生成器的种子以确保实验可重复
    
    参数:
        seed (int): 随机种子 (默认: 42)
        deterministic (bool): 是否启用PyTorch的确定性模式 (可能影响性能)
    
    影响范围:
        - Python内置random模块
        - NumPy
        - PyTorch (CPU/CUDA)
        - 环境变量PYTHONHASHSEED
        - CUDA卷积优化基准
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # 启用确定性模式 (可能降低性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # 保持性能优化但牺牲完全确定性
        torch.backends.cudnn.benchmark = True

    print(f"Set seed to {seed} with deterministic={deterministic}")