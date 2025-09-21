| File                         | Priority | Setting  |
|------------------------------|----------|----------|
| rarl_qwen3b_schedulerCos.sh | P0       | cosine scheduler  |
| rarl_qwen3b_layertype2.sh | P0       | 抽取层依次为35 30 25 20 15 10   |
| rarl_qwen3b_layertype3.sh | P0       | 抽取层依次为10 15 20 25 30 35   |
| rarl_qwen3b_layertype4.sh | P1       | 抽取层依次为10 20 30   |
| rarl_qwen3b_layertype1.sh | P1       | 抽取层依次为30 20 10   |

**代码有改动，需要在docker container 中git pull**