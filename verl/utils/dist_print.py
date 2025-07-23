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