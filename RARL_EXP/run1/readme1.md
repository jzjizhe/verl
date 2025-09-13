1. 对代码进行了修改，运行脚本前先**git pull**一下。
2. 本次主要是对regular loss weight进行调参
    
    - 调整weight：

    | File                       | Priority | Setting     |
    |----------------------------|----------|------------|
    | rarl_qwen3b_w1e-3.sh       | P0       | weight=1e-3 |
    | rarl_qwen3b_w8e-4.sh       | P0       | weight=8e-4 |
    | rarl_qwen3b_w7e-4.sh       | P0       | weight=7e-4 |
    | rarl_qwen3b_w5e-4.sh       | P0       | weight=5e-4 |
    | rarl_qwen3b_w6e-4.sh       | P1       | weight=6e-4 |
    | rarl_qwen3b_w4e-4.sh       | P2       | weight=4e-4 |

    - 对weight加入scheduler：

    | File                                 | Priority | Setting                       |
    |-------------------------------------|----------|-------------------------------|
    | rarl_qwen3b_schedulerCos.sh          | P0       | cosine scheduler              |
    | rarl_qwen3b_schedulerCosMin0.5.sh   | P0       | cosine scheduler+eta_min=0.5 |
    | rarl_qwen3b_schedulerfast.sh         | P1       | fast decay scheduler+eta_min=0.5 |
    | rarl_qwen3b_schedulerReverseCosMin0.5.sh | P2  | reverse cosine scheduler+eta min=0.5 |

