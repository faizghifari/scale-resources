"""Shared Unsloth FastModel LoRA load/train helpers for CPT and SFT scripts.

Used by scaleres.training.cpt_unsloth and sft_unsloth, which otherwise each
reimplemented a ~90%-identical load_model/train() pair (differing only in
LoRA target modules and a handful of training hyperparameters).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from unsloth import FastModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
CPT_EXTRA_TARGET_MODULES: List[str] = ["embed_tokens", "lm_head"]


def load_lora_model(
    model_id: str,
    device_id: int,
    *,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """Load a base model in 4-bit and wrap it with a LoRA adapter via Unsloth."""

    model, tokenizer = FastModel.from_pretrained(
        device_map=torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        ),
        model_name=model_id,
        tie_word_embeddings=False,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=os.getenv("HF_TOKEN"),
    )

    model = FastModel.get_peft_model(
        model,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    return model, tokenizer


@dataclass
class UnslothRunConfig:
    num_train_epochs: float
    learning_rate: float
    lr_scheduler_type: str
    embedding_learning_rate: Optional[float] = None


def train_lora(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    max_seq_length: int,
    run_name: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    save_steps: int,
    run_cfg: UnslothRunConfig,
) -> None:
    """Run an Unsloth LoRA training loop and save the adapter + tokenizer."""

    ta_kwargs = dict(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=run_cfg.num_train_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=run_cfg.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type=run_cfg.lr_scheduler_type,
        seed=3407,
        output_dir=output_dir,
        report_to="wandb",
        run_name=run_name,
    )
    if run_cfg.embedding_learning_rate is not None:
        ta_kwargs["embedding_learning_rate"] = run_cfg.embedding_learning_rate

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(**ta_kwargs),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    model.config.use_cache = False

    print("Training...")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()
