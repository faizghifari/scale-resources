#!/usr/bin/env python
"""Continued pretraining (CPT) on a base model via Unsloth LoRA."""

import argparse
import os

import wandb
from datasets import load_from_disk
from dotenv import load_dotenv

from scaleres.common.unsloth_common import (
    CPT_EXTRA_TARGET_MODULES,
    DEFAULT_TARGET_MODULES,
    UnslothRunConfig,
    load_lora_model,
    train_lora,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", help="path to the output directory", required=True, type=str
    )
    parser.add_argument(
        "--dataset_dir", help="path to the dataset directory", default=None, type=str
    )
    parser.add_argument("--model_id", help="model id", default=None, type=str)
    parser.add_argument("--max_length", help="maximum length", default=8192, type=int)
    parser.add_argument("--batch_size", help="batch size", default=1, type=int)
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="gradient accumulation steps",
        default=1,
        type=int,
    )
    parser.add_argument("--save_steps", help="save steps", default=1000, type=int)
    parser.add_argument("--project_name", help="project name", default=None, type=str)
    parser.add_argument("--run_name", help="run name", default=None, type=str)
    parser.add_argument("--device_id", help="device id", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    model, tokenizer = load_lora_model(
        args.model_id,
        args.device_id,
        target_modules=DEFAULT_TARGET_MODULES + CPT_EXTRA_TARGET_MODULES,
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
            num_train_epochs=1,
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            embedding_learning_rate=1e-5,
        ),
    )
    wandb.finish()


if __name__ == "__main__":
    main()
