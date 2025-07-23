
set -x
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_API_KEY='8b11eae0574f67497f2d6ff39806832bcb1ec92b'
export WANDB_MODE=offline
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# export CUDA_VISIBLE_DEVICES=0,1,2,3
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=/home/hhzhang/improve/verl/datasets/math/train.parquet
math_test_path=/home/hhzhang/improve/verl/datasets/math500/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"
model_path="/data2/data/qwen/Qwen2.5-1.5B-Instruct"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.log_val_generations=1 \
    trainer.project_name='verl_grpo_example_math' \
    trainer.experiment_name='qwen2dot5_1dot5b_instruct_function_demo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.total_epochs=5 \
    trainer.save_root=/data1/hhzhang/jizhe/verl $@

#   trainer.max_actor_ckpt_to_keep: null
#   default_local_dir: ${trainer.save_root}/${trainer.project_name}/${trainer.experiment_name}/checkpoints
#   default_local_epoch_dir: ${trainer.save_root}/${trainer.project_name}/${trainer.experiment_name}/epochs