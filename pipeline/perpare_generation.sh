#!/usr/bin/env bash

workdir=$(cd $(dirname $0); pwd)
echo $workdir

OLD_STEP=global_step4000
STEP=global_step36000
checkpoint_prefix=bloom-7b1-lark_sum-full

cd /mnt/bn/larkai-fr-gpt/
mkdir 7b1_merge_20230119
echo $STEP > latest
cd 7b1_merge_20230119
mv $OLD_STEP $STEP
mkdir $STEP


cd $STEP

echo "remove old layer checkpoint"
rm layer_*
echo "copy finetune 0"
/bin/cp -rf /mnt/bn/larkai-fr-gpt/$checkpoint_prefix_0/$STEP/layer_* .
echo "copy finetune 1"
/bin/cp -rf /mnt/bn/larkai-fr-gpt/output_dir/$checkpoint_prefix_1/$STEP/layer_* .
# echo "copy zero"
# /bin/cp -rf /mnt/bn/larkai-fr-gpt/models/bloom-7b1-optimizer-states/global_step337500/zero_pp* .

cd /mnt/bn/larkai-fr-gpt/
mkdir 7b1_megatron_20230119

cd $workdir/Megatron-DeepSpeed

echo "deepspeed to megatron"

export CMD=" \
    python3 tools/convert_checkpoint/deepspeed_to_megatron.py  \
		--input_folder /mnt/bn/larkai-fr-gpt/7b1_merge_20230119/$STEP \
		--output_folder /mnt/bn/larkai-fr-gpt/7b1_megatron_20230119
    "

echo $CMD
$CMD
