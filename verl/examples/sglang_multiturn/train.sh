cd SandboxFusion/
conda activate sandbox-runtime
make run-online PORT=8010  &
make run-online PORT=8020  &
make run-online PORT=8030  &
make run-online PORT=8040  &
make run-online PORT=8050  &
make run-online PORT=8060  &
make run-online PORT=8070  &
make run-online PORT=8080  &

cd ../verl
conda activate verl



set -x


export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_GDR_LEVEL=4
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_TC=186
export NCCL_NVLS_ENABLE=0
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TIMEOUT=22 
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA=^=mlx5_3,mlx5_4,mlx5_5,mlx5_bond_0
ulimit -l unlimited

nnodes=$PET_NNODES
current_rank=$PET_NODE_RANK
master_addr=$MASTER_ADDR
master_port=$MASTER_PORT


export WANDB_MODE=offline


PROJECT_DIR="$(pwd)"
policy_path=models/Qwen3-Coder-30B-A3B-Instruct
rollout_batch_size=32
n_samples_per_prompts=8
episode=2
temperature=1.0
batch_size=32
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
max_gen_length=10240
max_assistant_turns=20
SP_SIZE=1
TP_SIZE=8
project_name='sr-scientist'
run_name=qwen3-coder-30B


if [ "$current_rank" = "0" ]; then
    ray start --head --port=6379 --num-gpus 8
elif [ "$current_rank" != "0" ]; then
    sleep 20
    ray start --address=${master_addr}:6379 --num-gpus 8 --block
fi

if [ "$current_rank" = "0" ]; then
LOG_DIR="log/${project_name}"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${run_name}_$(TZ='UTC-8' date "+%Y%m%d_%H%M%S").log"

python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=sr_scientist  \
    algorithm.adv_estimator=grpo \
    data.train_files=../data/training/train.parquet \
    data.val_files=../data/training/test.parquet \
    data.train_batch_size=$rollout_batch_size \
    data.max_prompt_length=2560 \
    data.max_response_length=$max_gen_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$policy_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_call_parser="qwen3_coder" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns  \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompts \
    actor_rollout_ref.rollout.over_sample_rate=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${run_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/tool_config.yaml" \
    trainer.total_epochs=$episode $@ 2>&1 | tee ${LOG_FILE}
fi