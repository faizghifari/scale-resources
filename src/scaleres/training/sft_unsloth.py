#!/usr/bin/env python
"""Supervised fine-tuning (SFT / instruction-tuning) via Unsloth LoRA."""

import argparse
import os

import wandb
from datasets import load_from_disk
from dotenv import load_dotenv

from scaleres.common.unsloth_common import (
    DEFAULT_TARGET_MODULES,
    UnslothRunConfig,
    load_lora_model,
    train_lora,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--dataset_dir", default=None, type=str)
    parser.add_argument("--model_id", default=None, type=str)
    parser.add_argument("--max_length", default=8192, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--project_name", default=None, type=str)
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--device_id", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    model, tokenizer = load_lora_model(
        args.model_id, args.device_id, target_modules=DEFAULT_TARGET_MODULES
    )
    dataset = load_from_disk(args.dataset_dir)
    train_lora(
        model,
        tokenizer,
        dataset,
        args.output_dir,
        args.max_length,
        args.run_name,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.save_steps,
        run_cfg=UnslothRunConfig(
            num_train_epochs=3,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            embedding_learning_rate=0,
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    main()
