- 减小了dynamic batch size的max tokens

| File                         | Priority | Setting  |
|------------------------------|----------|----------|
| rarl_qwen3b_baseline4node.sh | P0       | 4 node   |
| rarl_qwen3b_baseline2node.sh | P0       | 2 node   |
| rarl_qwen3b_baseline1node.sh | P0       | 1 node   |

调整regular loss weight
| File                         | Priority | Setting  |
|------------------------------|----------|----------|
| rarl_qwen3b_w1e-3.sh | P0       | loss weight=1e-3   |
| rarl_qwen3b_w8e-4.sh | P0       | loss weight=8e-4   |
| rarl_qwen3b_w7e-4.sh | P0       | loss weight=7e-4   |
| rarl_qwen3b_w6e-4.sh | P1       | loss weight=6e-4   |
| rarl_qwen3b_w9e-4.sh | P1       | loss weight=9e-4   |

调整weight schduler

| File                         | Priority | Setting  |
|------------------------------|----------|----------|
| rarl_qwen3b_schedulerCos.sh | P0       | cosine scheduler   |
| rarl_qwen3b_schedulerCosMin0.5.sh | P0       | cosine scheduler+eta_min=0.5   |
| rarl_qwen3b_schedulerFastMin0.5.sh | P1       |    fast decay scheduler+eta_min=0.5 |
| rarl_qwen3b_schedulerReverseCosMin0.5.sh | P1       | reverse cosine scheduler+eta min=0.5   |
