TOKENIZER_PATH=bigscience/bloomz-7b1
DATA_ROOT=/mnt/bd/zenithbloomvol/data/IE_data_v3
# 会在OUTPUT文件夹下生成四个文件：
# IE_data_v3_inputs_document.bin
# IE_data_v3_inputs_document.idx
# IE_data_v3_targets_document.bin
# IE_data_v3_targets_document.idx

DATA_PATH=$DATA_ROOT/UIE_train.jsonl
OUTPUT=$DATA_ROOT/train   
python3 Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 32
python3 Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --prepend-space \
    --workers 32

DATA_PATH=$DATA_ROOT/UIE_dev.jsonl
OUTPUT=$DATA_ROOT/valid
python3 Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 32
python3 Megatron-DeepSpeed/tools/preprocess_data.py \
    --input $DATA_PATH \
    --output-prefix $OUTPUT \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --prepend-space \
    --workers 32