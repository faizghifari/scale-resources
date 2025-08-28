#!/usr/bin/env python
"""
train.py - Train a small LLaMA/Mistral-architecture LM from scratch on Balinese/Cirebonese datasets (fp16).

Key features:
- From-scratch initialization (random weights) with configurable small model size to fit ~10GB VRAM
- Architectures supported: LLaMA (default), Mistral
- Data: Load multiple datasets saved via save_to_disk under dataset/cpt/*
- Normalizes to a single 'text' column, tokenizes, and packs into fixed-length blocks for causal LM
- Pure fp16 (no bf16), gradient checkpointing, low batch sizes for memory efficiency

First, train a tokenizer using the dedicated script:
    python train_tokenizer.py \\
        --train_dirs dataset/cpt/bali_hq_200k,dataset/cpt/cbn_hq_2k \\
        --output_dir models/BaliCirebonese-Tokenizer --vocab_size 32000

Example usage (adjust paths as needed):
    python train.py \
        --train_dirs dataset/cpt/bali_hq_200k,dataset/cpt/bali_filtered_bt-85,dataset/cpt/cbn_hq_2k \
        --val_dirs dataset/cpt/bali_valid_hq_5000,dataset/cpt/cbn_valid_hq_500 \
        --output_dir models/BaliCirebonese-SmallLM --tokenizer_path_or_id models/BaliCirebonese-Tokenizer \
        --arch llama --dim 512 --n_layers 12 --n_heads 8 --seq_len 512 \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --fp16

Requirements:
    pip install transformers datasets accelerate
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.trainer_utils import get_last_checkpoint
import torch
import warnings

# no direct tokenizers usage to keep compatibility across versions

# Env/Warnings hygiene
# Avoid tokenizers parallelism fork warning and potential deadlocks when DataLoader uses workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Hide FutureWarning from torch about cpu autocast inside checkpointing (not actionable here).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small LLaMA/Mistral LM from scratch (fp16)"
    )
    parser.add_argument(
        "--tokenizer_path_or_id",
        type=str,
        required=True,
        help="Path to a local tokenizer directory or a Hugging Face Hub repository ID.",
    )
    parser.add_argument(
        "--train_dirs",
        type=str,
        default="dataset/cpt/bali_hq_200k,dataset/cpt/cbn_hq_2k",
        help="Comma-separated list of dataset save_to_disk directories for training.",
    )
    parser.add_argument(
        "--val_dirs",
        type=str,
        default="dataset/cpt/bali_valid_hq_5000,dataset/cpt/cbn_valid_hq_500",
        help="Comma-separated list of dataset save_to_disk directories for validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/SmallLLM-TinyLlama",
        help="Where to save outputs",
    )

    # Tokenization / packing
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="Sequence length for packing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training hyperparams (choose conservative defaults for 10GB VRAM)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument(
        "--save_fraction",
        type=float,
        default=None,
        help="If set (e.g., 0.1), save a checkpoint every given fraction of an epoch.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name (overrides env if set)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity/org (optional)",
    )

    # Model arch & size (from scratch)
    parser.add_argument(
        "--arch",
        type=str,
        default="llama",
        choices=["llama", "mistral"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--dim", type=int, default=768, help="Hidden size (model dimension)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=16, help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_kv_heads",
        type=int,
        default=None,
        help="Number of key/value heads (defaults to n_heads)",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="FFN intermediate size (defaults ~ 4*dim)",
    )
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")

    # Misc
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 (disabled by default)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use float16 mixed precision (default)",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing (uses more VRAM, faster)",
    )
    parser.add_argument(
        "--report_to", type=str, default="none", help="wandb/tensorboard/none"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        default=False,
        help="If set, resume from the last checkpoint found in output_dir.",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        default=False,
        help="Load model weights from the last checkpoint (no optimizer/scheduler state).",
    )
    parser.add_argument(
        "--init_from_path_or_id",
        type=str,
        default=None,
        help="Initialize model weights from a local model directory or HF model repo (ignores optimizer).",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=2, help="Early stopping patience in evaluation steps (patience rounds). Set 0 to disable."
    )
    parser.add_argument(
        "--early_stopping_threshold", type=float, default=0.0, help="Minimum improvement to qualify as better."
    )
    parser.add_argument(
        "--disable_early_stopping", action="store_true", default=False, help="Disable early stopping callback."
    )

    # Hub push options
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Push model to the Hugging Face Hub at the end."
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None, help="Target HF model repo id, e.g., 'user/repo'."
    )
    parser.add_argument(
        "--hub_private", action="store_true", default=True, help="Create/use private repo on the Hub."
    )
    parser.add_argument(
        "--hub_strategy", type=str, default="end", help="When to push to hub: 'end' or 'checkpoint' (if supported)."
    )

    return parser.parse_args()


def _as_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, DatasetDict):
        return concatenate_datasets([v for v in obj.values()])
    return obj


def ensure_text_column(ds: Dataset) -> Dataset:
    """Normalize dataset to have a single 'text' column.

    Preference order:
    - If 'text' exists, keep it and drop other columns.
    - Else, if one of ('balinese', 'cirebonese', 'indonesian') exists, keep that and rename to 'text'.
    - Else, if only one column exists, rename it to 'text'.
    - Otherwise, raise an error.
    """
    cols = list(ds.column_names)
    if "text" in cols:
        remove_cols = [c for c in cols if c != "text"]
        return ds.remove_columns(remove_cols) if remove_cols else ds
    for c in ("balinese", "cirebonese", "indonesian"):
        if c in cols:
            tmp = (
                ds.remove_columns([x for x in cols if x != c]) if len(cols) > 1 else ds
            )
            return tmp.rename_column(c, "text") if c != "text" else tmp
    if len(cols) == 1:
        only = cols[0]
        return ds.rename_column(only, "text") if only != "text" else ds
    raise ValueError(f"No suitable text column in dataset with columns: {cols}")


def load_concat_datasets(dir_list: List[str], split_name: str) -> Dataset:
    datasets = []
    for d in dir_list:
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] {split_name} dataset dir not found: {d}")
            continue
        try:
            ds_any = load_from_disk(d)
            ds = _as_dataset(ds_any)
            ds = ensure_text_column(ds)
            # Drop empty/None lines
            ds = ds.filter(
                lambda x: x["text"] is not None and str(x["text"]).strip() != ""
            )
            datasets.append(ds)
            print(f"Loaded {split_name} from {d}: {len(ds):,} rows")
        except Exception as e:
            print(f"[ERROR] Failed to load {d}: {e}")
    if not datasets:
        raise RuntimeError(f"No {split_name} datasets loaded from: {dir_list}")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def load_tokenizer(
    args: argparse.Namespace, train_ds: Dataset
) -> PreTrainedTokenizerFast:
    """Loads a tokenizer from a local path or Hugging Face Hub.

    Args:
        args: Command-line arguments.
        train_ds: The training dataset (unused, kept for signature consistency).

    Returns:
        The loaded fast tokenizer.
    """
    path_or_id = args.tokenizer_path_or_id
    print(f"Loading tokenizer from: {path_or_id}")
    try:
        # from_pretrained handles both local paths and hub IDs
        tokenizer = AutoTokenizer.from_pretrained(path_or_id, use_fast=True)
        return tokenizer  # type: ignore
    except Exception as e:
        print(
            f"[ERROR] Failed to load tokenizer from '{path_or_id}': {e}",
            file=sys.stderr,
        )
        raise


def build_model(args: argparse.Namespace, tokenizer: PreTrainedTokenizerFast):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_kv = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
    inter = (
        args.intermediate_size
        if args.intermediate_size is not None
        else int(4 * args.dim)
    )

    if args.arch == "llama":
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.dim,
            intermediate_size=inter,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            num_key_value_heads=n_kv,
            max_position_embeddings=args.seq_len,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
            use_cache=False,
        )
    else:  # mistral
        config = MistralConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.dim,
            intermediate_size=inter,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            num_key_value_heads=n_kv,
            max_position_embeddings=args.seq_len,
            rope_theta=args.rope_theta,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
            use_cache=False,
        )

    model = AutoModelForCausalLM.from_config(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


# No shim needed with up-to-date accelerate/transformers


def tokenize_and_group(tokenizer: Any, ds: Dataset, seq_len: int) -> Dataset:
    def tok(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

    tokenized = ds.map(
        tok,
        batched=True,
        remove_columns=ds.column_names,  # drop original columns; keep only tokenizer outputs
        desc="Tokenizing",
    )

    # Determine EOS id for boundary insertion
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        # Try common EOS token string
        try:
            eos_id = tokenizer.convert_tokens_to_ids("</s>")
        except Exception:
            eos_id = None
    if eos_id is None:
        raise ValueError(
            "Tokenizer must define an eos_token_id to insert EOS between documents."
        )

    # Concatenate with EOS separators and split into fixed-length blocks; build attention_mask to match
    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate input_ids with EOS between documents to mark boundaries
        concatenated_ids: List[int] = []
        if "input_ids" in examples:
            for i, ids in enumerate(examples["input_ids"]):
                if not ids:
                    continue
                concatenated_ids.extend(ids)
                # Insert EOS after each document
                concatenated_ids.append(eos_id)
        total_length = (len(concatenated_ids) // seq_len) * seq_len
        if total_length == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        input_blocks = [
            concatenated_ids[i : i + seq_len] for i in range(0, total_length, seq_len)
        ]
        attn_blocks = [[1] * seq_len for _ in range(len(input_blocks))]
        labels_blocks = [blk.copy() for blk in input_blocks]
        return {
            "input_ids": input_blocks,
            "attention_mask": attn_blocks,
            "labels": labels_blocks,
        }

    lm_ds = tokenized.map(group_texts, batched=True, desc="Grouping into blocks")
    return lm_ds


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dirs = [s.strip() for s in args.train_dirs.split(",") if s.strip()]
    val_dirs = [s.strip() for s in args.val_dirs.split(",") if s.strip()]

    print("Loading datasets...")
    train_raw = load_concat_datasets(train_dirs, split_name="train")
    val_raw = load_concat_datasets(val_dirs, split_name="validation")

    # Shuffle for mixing languages
    train_raw = train_raw.shuffle(seed=args.seed)
    val_raw = val_raw.shuffle(seed=args.seed)

    print(f"Train rows: {len(train_raw):,} | Val rows: {len(val_raw):,}")

    print("Preparing tokenizer...")
    tokenizer = load_tokenizer(args, train_raw)
    # Build or load model
    model = None
    last_ckpt = None
    if args.init_from_path_or_id:
        src = args.init_from_path_or_id
        print(f"Initializing model weights from: {src}")
        try:
            model = AutoModelForCausalLM.from_pretrained(src)
        except Exception as e:
            print(f"[ERROR] Failed to load model from {src}: {e}. Falling back to other init path.")
            model = None
    elif args.resume_weights_only:
        last_ckpt = (
            get_last_checkpoint(args.output_dir)
            if os.path.isdir(args.output_dir)
            else None
        )
        if last_ckpt is None:
            print(
                "[WARN] --resume_weights_only set but no checkpoint found; building a fresh model."
            )
        else:
            print(f"Loading model weights from checkpoint (weights only): {last_ckpt}")
            try:
                model = AutoModelForCausalLM.from_pretrained(last_ckpt)
            except Exception as e:
                print(
                    f"[ERROR] Failed to load model from {last_ckpt}: {e}. Falling back to fresh model."
                )
                model = None
    if model is None:
        print("Building model from scratch config...")
        model = build_model(args, tokenizer)

    # Configure Weights & Biases via args if requested
    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    # Tokenize and group
    print("Tokenizing and grouping datasets...")
    train_ds = tokenize_and_group(tokenizer, train_raw, args.seq_len)
    eval_ds = tokenize_and_group(tokenizer, val_raw, args.seq_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Enforce fp16-only per requirements
    if args.bf16:
        print("[WARN] bf16 requested but requirement is fp16-only; forcing bf16=False")
        args.bf16 = False
    if not args.fp16:
        print("[INFO] Enabling fp16 for training as per requirement")
        args.fp16 = True
    print("Precision: fp16")

    # Optionally convert save_fraction to save_steps (approximate)
    computed_save_steps = args.save_steps
    if args.save_fraction is not None:
        world_size = max(1, torch.cuda.device_count())
        eff_batch = (
            args.per_device_train_batch_size
            * world_size
            * args.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, int(np.ceil(len(train_ds) / max(1, eff_batch))))
        computed_save_steps = max(1, int(np.ceil(steps_per_epoch * args.save_fraction)))
        print(
            f"[save_fraction] steps_per_epoch≈{steps_per_epoch}, save every {computed_save_steps} steps"
        )
        # Ensure save_steps is a round multiple of eval_steps if loading best model at end
        if args.eval_steps and (computed_save_steps % args.eval_steps != 0):
            adjusted = (
                int(np.ceil(computed_save_steps / args.eval_steps)) * args.eval_steps
            )
            if adjusted != computed_save_steps:
                print(
                    f"[save_fraction] adjusting save_steps from {computed_save_steps} to {adjusted} to be a multiple of eval_steps={args.eval_steps}"
                )
                computed_save_steps = adjusted

    _ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=computed_save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        report_to=None if args.report_to == "none" else args.report_to,
        bf16=False,
        fp16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        run_name=args.run_name,
    )
    # Use new arg name to avoid deprecation warnings
    _ta_kwargs["eval_strategy"] = "steps"
    # Hub args
    if args.push_to_hub and args.hub_model_id:
        _ta_kwargs["push_to_hub"] = True
        _ta_kwargs["hub_model_id"] = args.hub_model_id
        _ta_kwargs["hub_private_repo"] = args.hub_private
        # hub_strategy key name varies by transformers version; set if available name
        _ta_kwargs["hub_strategy"] = args.hub_strategy
    training_args = TrainingArguments(**_ta_kwargs)

    callbacks = []
    if not args.disable_early_stopping and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    resume_arg: bool | str = False
    if args.resume_from_checkpoint and not args.resume_weights_only and not args.init_from_path_or_id:
        last_ckpt = (
            get_last_checkpoint(args.output_dir)
            if os.path.isdir(args.output_dir)
            else None
        )
        if last_ckpt is None:
            print("[INFO] No checkpoint found in output_dir; starting fresh.")
            resume_arg = False
        else:
            print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
            resume_arg = last_ckpt

    train_result = trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model()  # Save model config + weights
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Evaluating...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Optionally push to hub at end
    if getattr(training_args, "push_to_hub", False) and args.hub_model_id:
        try:
            trainer.push_to_hub()
            print(f"Pushed model to Hub: {args.hub_model_id}")
        except Exception as e:
            print(f"[WARN] push_to_hub failed: {e}")

    print("Done. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
