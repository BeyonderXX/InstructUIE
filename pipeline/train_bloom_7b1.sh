#!/usr/bin/env bash
set -e
export BYTED_TORCH_FX=O0
export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_LEGACY_FILESYSTEM=1
export NCCL_DEBUG=INFO
workdir=$(cd $(dirname $0); pwd)
echo $workdir
cd $workdir/Megatron-DeepSpeed
cd megatron/data
make 
cd ../../

BD=/mnt/bd/zenithbloomvol

# CHECKPOINT_PATH=/mnt/bn/larkai-fr-gpt/models/bloom-7b1-optimizer-states
CHECKPOINT_PATH=$BD/output/bloomz-7b1/checkpoint-6000
SAVE_CHECKPOINT_PATH=$BD/output/bloomz-7b1-pipe
TENSORBOARD_PATH=$BD/output/tensorboard
TRAIN_DATA_CONFIG=/mnt/bd/zenithbloomvol/data/IE_data_v3/data_config/train.txt
VALID_DATA_CONFIG=/mnt/bd/zenithbloomvol/data/IE_data_v3/data_config/valid.txt

export TRANSFORMERS_CACHE=$BD/huggingface
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

N_GPUS=${ARNOLD_WORKER_GPU}
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TP_SIZE=1
PP_SIZE=8

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048

SAVE_INTERVAL=1

TRAIN_SAMPLES=256   #总共会训练的条目数
# train-iters=3

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --lr-decay-style constant \
    --lr-warmup-samples 0 \
    --clip-grad 1.0 \
    --weight-decay 1e-4 \
    --no-load-optim \
    --reset-progress \
    --norm-target-loss \
    "

# EXIT_OPTS=" \
#     --exit-duration-in-mins 5990 \
#     "

GPT_ARGS=" \
    --pp-partition-method type:transformer|embedding \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --clip-grad 1.0 \
    --embed-layernorm \
    --no-load-optim \
    --reset-progress \
    --finetune \
    --checkpoint-activations \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path  bigscience/bloomz-7b1 \
    --abort-on-unmet-fused-kernel-constraints \
    --pad-vocab-size-to 250880 \
    --init-method-std 0.0048 \
    --fp16 \
    --partition-activations \
    --position-embedding-type alibi \
    --seed 42 \
    $OPTIMIZER_ARGS \
    "
        # --finetune \

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 125 \
    --eval-iters 10 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

DATA_ARGS=" \
    --save $SAVE_CHECKPOINT_PATH \
    --train-weighted-split-paths-path $TRAIN_DATA_CONFIG \
    --valid-weighted-split-paths-path $VALID_DATA_CONFIG \
    --tensorboard-dir $TENSORBOARD_PATH \
    --dataloader-type single \
    --data-impl mmap \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

ZERO_STAGE=1

config_json="./ds_config.json"
  

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

MASTER_ADDR=${METIS_WORKER_0_HOST}
MASTER_PORT=${METIS_WORKER_0_PORT}

export LAUNCHER="python3 -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes ${ARNOLD_WORKER_NUM} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --rdzv_conf read_timeout=3600000 \
    --tee 3 \
    "
    
export CMD=" \
    $LAUNCHER finetune_t0.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

$CMD