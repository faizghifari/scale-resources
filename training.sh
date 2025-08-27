#!/usr/bin/env bash
set -euo pipefail

# Train a Balinese-only small LM from scratch using train.py
# - Trains a language-specific tokenizer (32k default)
# - Uses Balinese train/val datasets under dataset/cpt

CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dirs dataset/cpt/bali_hq_200k\
    --val_dirs dataset/cpt/bali_valid_hq_5000 \
    --output_dir models/Balinese-SmallLM \
    --train_tokenizer \
    --report_to wandb \
    --wandb_project BaliLM \
    --run_name bali-32k-synth_all-hq_2x-150M