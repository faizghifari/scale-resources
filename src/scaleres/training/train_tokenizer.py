#!/usr/bin/env python
"""
train_tokenizer.py - Train a BPE tokenizer from scratch on text datasets.

This script takes one or more text datasets, normalizes them to a 'text' column,
and trains a new SentencePiece BPE tokenizer using the Hugging Face tokenizers
and transformers libraries.

The trained tokenizer is saved to a specified directory and can be used for
training a new language model.

Example usage:
    python -m scaleres.training.train_tokenizer \\
        --train_dirs dataset/cpt/ban_hq_200k,dataset/cpt/cbn_hq_2k \\
        --output_dir models/BaliCirebonese-Tokenizer \\
        --vocab_size 32000 \\
        --push_to_hub --hf_repo_id="username/my-tokenizer"
"""

from __future__ import annotations

import argparse
import os
import sys

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from scaleres.common.hf_dataset import load_concat_datasets

# Env hygiene
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a new tokenizer from text datasets"
    )
    parser.add_argument(
        "--train_dirs",
        type=str,
        required=True,
        help="Comma-separated list of dataset save_to_disk directories for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Tokenizer vocab size.",
    )
    parser.add_argument(
        "--tokenizer_template_id",
        type=str,
        default="gpt2",
        help="Base fast tokenizer to use as a template for train_new_from_iterator (e.g., gpt2).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained tokenizer to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID to push the tokenizer to (e.g., 'username/my-tokenizer'). Required if --push_to_hub is set.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face Hub token for authentication. Can also be set via HF_TOKEN env var or `huggingface-cli login`.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling."
    )
    args = parser.parse_args()
    if args.push_to_hub and not args.hf_repo_id:
        parser.error("--hf_repo_id is required when --push_to_hub is set.")
    return args


def train_tokenizer(args: argparse.Namespace, train_ds) -> PreTrainedTokenizerFast:
    print(
        f"Training tokenizer on {len(train_ds):,} examples | vocab_size={args.vocab_size}"
    )
    try:
        template_id = args.tokenizer_template_id or "gpt2"
        print(f"Using tokenizer template: {template_id}")
        base_tok = AutoTokenizer.from_pretrained(template_id, use_fast=True)

        def text_iter():
            for t in train_ds["text"]:
                if t is None:
                    continue
                s = str(t).strip()
                if s:
                    yield s

        new_tok = base_tok.train_new_from_iterator(text_iter(), args.vocab_size)

        # Ensure special tokens exist
        special_dict = {}
        if new_tok.unk_token is None:
            special_dict["unk_token"] = "<unk>"
        if new_tok.bos_token is None:
            special_dict["bos_token"] = "<s>"
        if new_tok.eos_token is None:
            special_dict["eos_token"] = "</s>"
        if new_tok.pad_token is None:
            special_dict["pad_token"] = "<pad>"
        if special_dict:
            new_tok.add_special_tokens(special_dict)

        # Set a very large model_max_length to avoid warnings. The actual sequence length
        # is determined by the model's configuration and the training script.
        new_tok.model_max_length = 10_000_000
        print(f"Set tokenizer's model_max_length to {new_tok.model_max_length}")

        os.makedirs(args.output_dir, exist_ok=True)
        new_tok.save_pretrained(args.output_dir)
        print(f"Saved trained tokenizer -> {args.output_dir}")

        if args.push_to_hub:
            print(f"Pushing tokenizer to Hugging Face Hub: {args.hf_repo_id}")
            try:
                new_tok.push_to_hub(
                    repo_id=args.hf_repo_id,
                    private=True,
                    token=args.hf_token,
                    commit_message="Upload tokenizer",
                )
                print("Successfully pushed tokenizer to the Hub.")
            except Exception as e:
                print(f"[ERROR] Failed to push tokenizer to Hub: {e}", file=sys.stderr)

        return new_tok
    except Exception as e:
        print(f"[ERROR] Tokenizer training failed: {e}", file=sys.stderr)
        raise


def main():
    args = parse_args()
    train_dirs = [s.strip() for s in args.train_dirs.split(",") if s.strip()]
    print("Loading datasets for tokenizer training...")
    train_raw = load_concat_datasets(train_dirs, split_name="train", warn_to_stderr=True)
    train_raw = train_raw.shuffle(seed=args.seed)
    print(f"Total rows for training: {len(train_raw):,}")
    train_tokenizer(args, train_raw)
    print("Tokenizer training complete.")


if __name__ == "__main__":
    main()
