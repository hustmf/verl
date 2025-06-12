set -x
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
xray check-hpn


# /share/zhuangwenhao/models/Qwen2.5-7B
export VLLM_ATTENTION_BACKEND=XFORMERS
export CKPT_PATH="/data/verl-rl/ckpt/Qwen2.5-32B"
export PROJECT_NAME='CODE_MATH_R1_Zero'
export EXPERIMENT_NAME=32B_code_math_22nodes_0320_final
export WANDB_API_KEY=d0433bdb2160b7f43a580b01901abc19935ea295
LOG_DIR="logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"
# scripts/code+math/7B-code-easy-hard.sh
# 判断路径是否存在
if [ ! -d "$LOG_DIR" ]; then
  # 路径不存在，创建路径
  mkdir -p "$LOG_DIR"
  echo "Directory $LOG_DIR created."
else
  echo "Directory $LOG_DIR already exists."
fi

# 创建日志文件
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
touch "$LOG_FILE"
echo "Log file $LOG_FILE created."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/verl-rl/data/deepscaler_shuffle/train-new.parquet \
    data.val_files=/data/verl-rl/data/deepscaler_shuffle/train-new.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=/data/verl-rl/ckpt/Qwen2.5-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.logprobs=1 \
    actor_rollout_ref.rollout.max_tokens=2048 \
    actor_rollout_ref.rollout.best_of=8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.min_p=0.0 \
    actor_rollout_ref.rollout.detokenize=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    reward_model.reward_manager="mindspeed_rl" \
    trainer.critic_warmup=0 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.logger=['console'] \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 2>&1 | tee $LOG_FILE

