1. 对代码进行了修改，运行脚本前先**git pull**一下。
2. 本次主要是对regular loss weight进行调参
    

    | File                       | Priority | Setting     |
    |----------------------------|----------|------------|
    | rarl_qwen3b_schedulerCos.sh          | P0       | cosine scheduler              |
    | rarl_qwen3b_schedulerCosMin0.5.sh   | P0       | cosine scheduler+eta_min=0.5 |
    | rarl_qwen3b_schedulerfast.sh         | P0       | fast decay scheduler+eta_min=0.5 |
    | rarl_qwen3b_schedulerReverseCosMin0.5.sh | P0  | reverse cosine scheduler+eta min=0.5 |


