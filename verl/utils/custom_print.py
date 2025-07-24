from termcolor import colored
import os

def is_rank_zero():
    # 兼容 torch.distributed 和 deepspeed
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True  # 单机单卡时
def rank_zero_print(msg):
    if not is_rank_zero():
        return
    width = 50
    print("*" * width)
    if isinstance(msg, str):
        lines = msg.splitlines()
    else:
        lines = [str(msg)]
    for line in lines:
        print("*" + line.center(width - 2) + "*")
    print("*" * width)

def format_print(msg):
    width = 50
    print("*" * width)
    if isinstance(msg, str):
        lines = msg.splitlines()
    else:
        lines = [str(msg)]
    for line in lines:
        print("*" + line.center(width - 2) + "*")
    print("*" * width)
# 更美观的实现
# from rich.console import Console
# from rich.panel import Panel

# console = Console()
# console.print(Panel("hello world\n你好 世界", width=50, title="Info", expand=False))