#!/usr/bin/env python
"""Continued pretraining via LoRA, on a single consumer GPU, without Unsloth.

Written for the Cirebonese generator experiment: every generator tested writes
Javanese because none has real Cirebonese exposure (F61-F65), and no prompt
lever fixes that -- the best recipe SUPPRESSES Javanese (2.04 -> 0.42 marker
rate) but never gets the model to produce Cirebonese-exclusive vocabulary it
does not have. Putting the language into the weights is the only untried route.

Deliberately plain peft rather than the repo's Unsloth path: Unsloth pins torch
and transformers hard, this stack is pinned for other reasons, and P0.4 already
cost a day to unbreak. bf16 LoRA without 4-bit quantisation also avoids adding
bitsandbytes. On a 16 GB card that budget is roughly:

    weights   4.02B x 2 bytes            ~8.0 GB
    LoRA r32 over 7 projections           ~0.13 GB
    AdamW fp32 states for adapters only   ~0.52 GB
    activations, checkpointing, seq 1024  ~1-2 GB

which leaves headroom. Full fine-tuning would need ~48 GB and is not the point:
with under 2M tokens of target text, LoRA is also the correct capacity.

    python -m scaleres.training.cpt_lora \\
        --base models/base/Qwen3-4B-Instruct-2507 \\
        --train dataset/cpt/cbn_train_disjoint dataset/cpt/cbn_tierB \\
        --out models/Qwen3-4B-cbn-lora
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batch", type=int, default=2,
                    help="HF defaults this to 8, which on a 16GB card allocates "
                         "a block big enough to fragment the allocator for the "
                         "rest of the run -- steps went 19s -> 7.7min at 98% "
                         "VRAM, a thrash rather than an OOM")
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to resume from")
    a = ap.parse_args()

    import torch
    from datasets import concatenate_datasets, load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    parts = []
    for p in a.train:
        d = load_from_disk(str(ROOT / p) if not Path(p).is_absolute() else p)
        if hasattr(d, "keys") and not hasattr(d, "column_names"):
            d = d[list(d.keys())[0]]
        parts.append(d.select_columns(["text"]))
        print(f"  {p}: {len(d):,} rows")
    ds = concatenate_datasets(parts).shuffle(seed=a.seed)
    print(f"pooled: {len(ds):,} rows")

    def tok_fn(b):
        return tok(b["text"], add_special_tokens=True)

    toked = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                   desc="tokenizing")

    # Pack into fixed blocks. Concatenating then slicing wastes nothing, which
    # matters when the whole corpus is under 2M tokens.
    def group(b):
        ids = []
        for x in b["input_ids"]:
            ids.extend(x)
        n = (len(ids) // a.seq_len) * a.seq_len
        blocks = [ids[i:i + a.seq_len] for i in range(0, n, a.seq_len)]
        return {"input_ids": blocks, "labels": [x[:] for x in blocks]}

    packed = toked.map(group, batched=True, batch_size=1000,
                       remove_columns=toked.column_names, desc="packing")
    ntok = len(packed) * a.seq_len
    print(f"packed: {len(packed):,} blocks of {a.seq_len} = {ntok/1e6:.2f}M tokens")

    split = packed.train_test_split(test_size=a.val_frac, seed=a.seed)
    train_ds, val_ds = split["train"], split["test"]

    # transformers 4.55 takes torch_dtype; `dtype` is a newer alias and raises
    # a TypeError from inside the model constructor on this pinned version.
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B "
          f"({100*trainable/total:.3f}%)")

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=a.eval_batch,
        eval_accumulation_steps=1,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, report_to=[], seed=a.seed,
        dataloader_num_workers=2, optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

    base_eval = trainer.evaluate()
    print(f"BASE eval loss {base_eval['eval_loss']:.4f} "
          f"ppl {math.exp(base_eval['eval_loss']):.1f}")

    trainer.train(resume_from_checkpoint=a.resume)
    final = trainer.evaluate()
    print(f"TUNED eval loss {final['eval_loss']:.4f} "
          f"ppl {math.exp(final['eval_loss']):.1f}")

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    summary = {
        "base": a.base, "train": a.train, "out": a.out,
        "blocks": len(packed), "seq_len": a.seq_len, "tokens": ntok,
        "lora": {"r": a.rank, "alpha": a.alpha, "dropout": a.dropout,
                 "trainable_M": round(trainable / 1e6, 2)},
        "lr": a.lr, "epochs": a.epochs,
        "eff_batch": a.batch * a.accum,
        "base_eval_loss": round(base_eval["eval_loss"], 4),
        "tuned_eval_loss": round(final["eval_loss"], 4),
        "base_ppl": round(math.exp(base_eval["eval_loss"]), 2),
        "tuned_ppl": round(math.exp(final["eval_loss"]), 2),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(a.out, "cpt_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
