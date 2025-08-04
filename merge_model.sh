#!/bin/bash
root=/data1/jzzhang/verl_results/Qwen2.5-1.5B
# 递归查找所有包含 checkpoints/global_step_*/actor 的目录
find $root -type d -path */model/global_step_*/actor | while read path; do
    if [ -f ${path}/model_world_size_*_rank_0.pt ]; then
    echo Processing $path
    python ./scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "${path}" \
        --target_dir "${path}" && rm -rf ${path}/model_world_size_*_rank_*.pt
    fi
done
# path=/data1/jzzhang/verl_results/Qwen2.5-3B/qwen3b_ep3_math_golden/model/global_step_10/actor
# python ./scripts/legacy_model_merger.py merge \
#     --backend fsdp \
#     --local_dir "${path}" \
#     --target_dir "${path}"