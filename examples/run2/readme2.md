
1. 使用不同layer的hidden state进行对齐
    | File                       | Priority | Setting     |
    |----------------------------|----------|------------|
    | rarl_qwen3b_layer4.sh       | P0       | layer=4 |
    | rarl_qwen3b_layer10.sh       | P0       | layer=10 |
    | rarl_qwen3b_layer27.sh       | P0       | layer=27 |
    | rarl_qwen3b_layer-1.sh       | P0       | layer=-1 |
    | rarl_qwen3b_layer18.sh       | P1       | layer=18 |

2. qwen7B baseline
    | File                       | Priority | Setting     |
    |----------------------------|----------|------------|
    | rarl_qwen7b_baseline.sh       | P0       | Qwen7B GRPO baseline, 8node*8gpu  |
    **Qwen7b的下载link**: https://huggingface.co/Qwen/Qwen-7B