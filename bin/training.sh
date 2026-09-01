#!/usr/bin/env bash
set -euo pipefail

# Train a Balinese-only small LM from scratch using scaleres.training.train
# - Uses a pre-trained tokenizer (train.py no longer trains one inline as of
#   commit 8b469ec; run bin/build_tokenizer.sh first if models/tokenizers_v6/ban_32k
#   does not match what you intend to train against -- this path is a best-guess
#   default, not necessarily what the original --train_tokenizer run used)
# - Uses Balinese train/val datasets under dataset/cpt

CUDA_VISIBLE_DEVICES=0 python -m scaleres.training.train \
    --train_dirs dataset/cpt/ban_hq_200k\
    --val_dirs dataset/cpt/ban_valid_hq_5000 \
    --output_dir models/Balinese-SmallLM \
    --tokenizer_path_or_id models/tokenizers_v6/ban_32k \
    --report_to wandb \
    --wandb_project BaliLM \
    --run_name bali-32k-synth_all-hq_2x-150M