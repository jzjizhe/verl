set -x
# aime24_test_path=/data0/jzzhang/verl/datasets/aime24/aime2024_math_format.parquet
# aime25_test_path=/data0/jzzhang/verl/datasets/aime25/aime2025_math_format.parquet
numina_test_path=/data0/jzzhang/AstirPair/RARL/test/numina_test.parquet
test_files="['$numina_test_path']"
model_path=/data1/jzzhang/verl_results/Qwen2.5-1.5B-Instruct/ep1_step200_baseline/model/global_step_25/actor
save_path=/data0/jzzhang/data
export CUDA_VISIBLE_DEVICES=1,2
python3 -m verl.trainer.main_generation_custom \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path="$test_files" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    data.result_path=${save_path}/evaluation.txt \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=-1 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
