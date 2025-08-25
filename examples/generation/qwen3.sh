
set -x
data_path=/home/hhzhang/improve/verl/datasets/math500/test.parquet
save_path=/data1/hhzhang/improve/verl_results/math500/test_llama8b_gen.parquet
# model_path=/data2/data/llama/Llama-3.1-8B
model_path=/data2/data/qwen/Qwen2.5-0.5B-Instruct
export CUDA_VISIBLE_DEVICES=9
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0 \
    rollout.top_k=-1 \
    rollout.top_p=1 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
