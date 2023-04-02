export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TOKENIZER_PATH="bloom_tokenizer"
python3 tools/preprocess_data.py \
    --input summary.jsonl \
    --output-prefix summary_data/document \
    --dataset-impl mmap \
    --json-key inputs \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --workers 8
python3 tools/preprocess_data.py \
    --input summary.jsonl \
    --output-prefix summary_data/document \
    --dataset-impl mmap \
    --json-key targets \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --append-eod \
    --prepend-space \
    --workers 8


# LANGS=(
# ak
# ar
# as
# bm
# bn
# ca
# code
# en
# es
# eu
# fon
# fr
# gu
# hi
# id
# ig
# ki
# kn
# lg
# ln
# ml
# mr
# ne
# nso
# ny
# or
# pa
# pt
# rn
# rw
# sn
# st
# sw
# ta
# te
# tn
# ts
# tum
# tw
# ur
# vi
# wo
# xh
# yo
# zh
# zu
# )
# # 
# DATA_PATH=/gpfswork/rech/six/commun/bigscience-training/jsonls/xp3cappedmixednewcodelong
# OUTPUT=/gpfswork/rech/six/commun/bigscience-training/xp3cappedmixednewcodelong

# mkdir -p $OUTPUT

# for val in {0..46}; do
#     LANG=${LANGS[$val]}
#     cd $DATA_PATH/$LANG
#     # Merge
#     cat *.jsonl > merged_dups_$LANG.jsonl
#     # Drop duplicates (~1G / 37G for en) + Shuffle
#     sort -u merged_dups_$LANG.jsonl | shuf > merged_$LANG.jsonl
#     cd $MEGATRON_DEEPSPEED_REPO
#     python tools/preprocess_data.py \
#         --input $DATA_PATH/$LANG/merged_$LANG.jsonl \
#         --output-prefix $OUTPUT/xp3_$LANG \
#         --dataset-impl mmap \
#         --json-key inputs \
#         --tokenizer-type PretrainedFromHF \
#         --tokenizer-name-or-path $TOKENIZER_PATH \
#         --workers 35
#     python tools/preprocess_data.py \
#         --input $DATA_PATH/$LANG/merged_$LANG.jsonl \
#         --output-prefix $OUTPUT/xp3_$LANG \
#         --dataset-impl mmap \
#         --json-key targets \
#         --tokenizer-type PretrainedFromHF \
#         --tokenizer-name-or-path $TOKENIZER_PATH \
#         --append-eod \
#         --prepend-space \
#         --workers 35
# done