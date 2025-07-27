
set -x

data_path=/data0/jzzhang/verl/datasets/math500/test.parquet
save_path=$HOME/data/gsm8k/deepseek_v2_lite_gen_test.parquet
model_path=/data1/jzzhang/verl_results/Qwen2.5-3B/qwen3b_ep3_math_golden/model/global_step_10/actor
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
