#!/usr/bin/env bash
set -euo pipefail

# This script trains a new BPE tokenizer from scratch and optionally pushes it to the Hugging Face Hub.
# The resulting tokenizer is saved to a local directory and can be used for pre-training a language model.

# --- Configuration ---
# Comma-separated list of dataset directories.
# These should be directories created by datasets' save_to_disk method.
TRAIN_DATA="DATA_PATH"

# Directory where the trained tokenizer will be saved locally.
OUTPUT_DIR="TOKENIZER_OUT_PATH"

# The desired vocabulary size for the tokenizer.
VOCAB_SIZE=32000

# The base tokenizer to use as a template. 'gpt2' is a good default.
TEMPLATE="gpt2"

# --- Hugging Face Hub Configuration ---
# Set to true to push the tokenizer to the Hub after training.
PUSH_TO_HUB=true
# The repository ID on the Hugging Face Hub (e.g., "username/my-tokenizer").
# Required if PUSH_TO_HUB is true.
HF_REPO_ID="REPO_ID"
# Optional: Hugging Face token. Can also be set via `huggingface-cli login` or the HF_TOKEN environment variable.
# If you want to pass it as an argument, uncomment and set the line below.
# HF_TOKEN_ARG="--hf_token your_token_here"

# --- Run Tokenizer Training ---
echo "Starting tokenizer training..."
echo "  - Training data:    ${TRAIN_DATA}"
echo "  - Output directory:   ${OUTPUT_DIR}"
echo "  - Vocab size:       ${VOCAB_SIZE}"

CMD_ARGS=(
    --train_dirs "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --vocab_size "${VOCAB_SIZE}" \
    --tokenizer_template_id "${TEMPLATE}" \
    --seed 42
)

if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    if [[ -z "${HF_REPO_ID}" ]]; then
        echo "Error: HF_REPO_ID must be set when PUSH_TO_HUB is true." >&2
        exit 1
    fi
    CMD_ARGS+=(--push_to_hub --hf_repo_id "${HF_REPO_ID}")
    echo "  - Push to Hub:      ${HF_REPO_ID}"
fi

# Add token argument if it's set
if [[ -n "${HF_TOKEN_ARG:-}" ]]; then
    CMD_ARGS+=(${HF_TOKEN_ARG})
fi

python train_tokenizer.py "${CMD_ARGS[@]}"

echo ""
echo "Tokenizer training complete."
if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    echo "Tokenizer saved to '${OUTPUT_DIR}' and pushed to '${HF_REPO_ID}'"
else
    echo "Tokenizer saved to '${OUTPUT_DIR}'"
fi