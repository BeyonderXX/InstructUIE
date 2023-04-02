#!/usr/bin/env bash
set -e
export BYTED_TORCH_FX=O0
export NCCL_IB_DISABLE=0 
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0
export TOKENIZERS_PARALLELISM=false
export TF_ENABLE_LEGACY_FILESYSTEM=1
workdir=$(cd $(dirname $0); pwd)
echo $workdir
cd $workdir/Megatron-DeepSpeed
cd megatron/data
make 
cd ../../

CHECKPOINT_PATH=/mnt/bn/larkai-fr-gpt/models/gpt2

VOCAB_FILE=/mnt/bn/larkai-fr-gpt/demo/llm_finetune-master/Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/mnt/bn/larkai-fr-gpt/demo/llm_finetune-master/Megatron-DeepSpeed/data/gpt2-merges.txt
DATA_PATH=/mnt/bn/larkai-fr-gpt/demo/llm_finetune-master/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=/mnt/bn/larkai-fr-gpt/demo/llm_finetune-master/Megatron-DeepSpeed/output_dir/tensorboard
SAVE_CHECKPOINT_PATH=/mnt/bn/larkai-fr-gpt/demo/llm_finetune-master/Megatron-DeepSpeed/output_dir/gpt2-finetune-demo_${METIS_TASK_INDEX}

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=2
NHIDDEN=8
NHEADS=2
SEQ_LEN=512
VOCAB_SIZE=50257

SAVE_INTERVAL=50

TRAIN_SAMPLES=10_000

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 12 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --embed-layernorm \
    --fp16 \
    --partition-activations \
    --seed 42 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    "

OUTPUT_ARGS=" \
    --exit-interval 300 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $SAVE_CHECKPOINT_PATH  \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --data-impl mmap \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

# DATA_ARGS=" \
#     --save $SAVE_CHECKPOINT_PATH \
#     --load $CHECKPOINT_PATH \
#     --train-weighted-split-paths-path xp3_train.txt \
#     --valid-weighted-split-paths-path xp3_validation.txt \
#     --tensorboard-dir $TENSORBOARD_PATH \
#     --dataloader-type single \
#     --data-impl mmap \
#     --tensorboard-queue-size 5 \
#     --log-timers-to-tensorboard \
#     --log-batch-size-to-tensorboard \
#     --log-validation-ppl-to-tensorboard \
#     --kill-switch-path /tmp/kill-switch \
#     "

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

# MASTER_ADDR=localhost
# MASTER_PORT=6777

MASTER_ADDR=${METIS_WORKER_0_HOST}
MASTER_PORT=${METIS_WORKER_0_PORT}

export LAUNCHER="python3 -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes ${ARNOLD_WORKER_NUM} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD