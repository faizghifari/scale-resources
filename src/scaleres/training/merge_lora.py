#!/usr/bin/env python
"""Merge a trained PEFT/LoRA adapter into its base model and save the result.

Renamed from the repo-root ``merge.py`` (moved here, next to the CPT/SFT
scripts that produce the adapters it merges) -- this is a post-training step,
not data preparation.
"""

import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)


def load_model(model_name: str):
    if "gemma-3" in model_name:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        )
        processor = AutoProcessor.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False,
        )
        processor = AutoTokenizer.from_pretrained(model_name)

    return model, processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge model and processor/tokenizer")
    parser.add_argument("--model_file", type=str, help="Path to model file", required=True)
    parser.add_argument("--lora_file", type=str, help="Path to lora file", required=True)
    parser.add_argument(
        "--output_dir", help="path to the output directory", required=True, type=str
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    peft_model_id = args.lora_file
    config = PeftConfig.from_pretrained(peft_model_id)

    model, processor = load_model(args.model_file)
    model = PeftModel.from_pretrained(model, peft_model_id)

    merged_model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)

    merged_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
