#!/usr/bin/env python
"""
eval_ppl.py - Compute perplexity of a trained causal LM on a validation/test dataset.

Inputs:
- --model_path_or_id: local model dir or HF repo id (will be loaded on GPU if available)
- --tokenizer_path_or_id: optional, default to model_path_or_id
- --data_dirs: comma-separated list of datasets saved with save_to_disk

Notes:
- Tokenization + packing matches train.py: insert EOS between documents, then pack into fixed-length blocks with labels=input_ids.
- Perplexity is computed from token-level average negative log-likelihood across all packed tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import default_data_collator

# Reduce tokenizers parallelism warning/noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity on a dataset")
    p.add_argument("--model_path_or_id", type=str, required=True)
    p.add_argument("--tokenizer_path_or_id", type=str, default=None)
    p.add_argument(
        "--data_dirs",
        type=str,
        required=True,
        help="Comma-separated list of dataset save_to_disk directories.",
    )
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--shuffle", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of rows before tokenization (for a quick check).",
    )
    return p.parse_args()


def _as_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, DatasetDict):
        return concatenate_datasets([v for v in obj.values()])
    return obj


def ensure_text_column(ds: Dataset) -> Dataset:
    cols = list(ds.column_names)
    if "text" in cols:
        drop = [c for c in cols if c != "text"]
        return ds.remove_columns(drop) if drop else ds
    for c in ("balinese", "cirebonese", "indonesian", "javanese", "content"):
        if c in cols:
            tmp = (
                ds.remove_columns([x for x in cols if x != c]) if len(cols) > 1 else ds
            )
            return tmp.rename_column(c, "text") if c != "text" else tmp
    if len(cols) == 1:
        only = cols[0]
        return ds.rename_column(only, "text") if only != "text" else ds
    raise ValueError(f"No suitable text column in dataset with columns: {cols}")


def load_concat_datasets(dir_list: List[str]) -> Dataset:
    datasets = []
    for d in dir_list:
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] dataset dir not found: {d}")
            continue
        try:
            ds_any = load_from_disk(d)
            ds = _as_dataset(ds_any)
            ds = ensure_text_column(ds)
            ds = ds.filter(
                lambda x: x["text"] is not None and str(x["text"]).strip() != ""
            )
            datasets.append(ds)
            print(f"Loaded from {d}: {len(ds):,} rows")
        except Exception as e:
            print(f"[ERROR] Failed to load {d}: {e}")
    if not datasets:
        raise RuntimeError(f"No datasets loaded from: {dir_list}")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def tokenize_and_group(tokenizer: Any, ds: Dataset, seq_len: int) -> Dataset:
    def tok(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

    tokenized = ds.map(
        tok,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        try:
            eos_id = tokenizer.convert_tokens_to_ids("</s>")
        except Exception:
            eos_id = None
    if eos_id is None:
        raise ValueError(
            "Tokenizer must define eos_token_id to insert EOS between documents."
        )

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated_ids: List[int] = []
        for ids in examples.get("input_ids", []):
            if not ids:
                continue
            concatenated_ids.extend(ids)
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


def _list_checkpoints(model_path: str) -> List[Dict[str, Any]]:
    """Return list of dicts with path and step for base model and its checkpoints (if any).

    For non-directory sources (e.g., HF repo id), returns just the source with step=None.
    """
    if not os.path.isdir(model_path):
        return [{"path": model_path, "step": None}]
    items: List[Dict[str, Any]] = []
    # Include base model directory as "final" (step=None)
    items.append({"path": model_path, "step": None})
    patt = re.compile(r"checkpoint-(\d+)$")
    try:
        for name in os.listdir(model_path):
            full = os.path.join(model_path, name)
            if not os.path.isdir(full):
                continue
            m = patt.search(name)
            if not m:
                continue
            step = int(m.group(1))
            # quick validity check
            if os.path.isfile(os.path.join(full, "config.json")):
                items.append({"path": full, "step": step})
    except Exception:
        pass
    # sort by step (None first for final, then ascending)
    items[1:] = sorted(
        [x for x in items[1:] if x["step"] is not None],
        key=lambda d: cast(int, d["step"]),
    )
    return items


def _model_name_from_path_or_id(src: str) -> str:
    if os.path.isdir(src):
        name = os.path.basename(os.path.normpath(src))
    else:
        name = src.split("/")[-1]
    # sanitize
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name or "model"


def compute_ppl_for_model(
    model_path: str, dl: DataLoader, device: torch.device
) -> Dict[str, Any]:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = cast(torch.nn.Module, model).to(device)  # type: ignore[call-arg]
    model.eval()
    total_tokens = 0
    total_nll = 0.0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype: Optional[torch.dtype] = torch.float16 if device_type == "cuda" else None
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                outputs = cast(Any, model)(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss  # mean over tokens in batch
                bsz, seqlen = input_ids.shape
                tokens = bsz * seqlen
                total_tokens += tokens
                total_nll += float(loss.item()) * tokens
    if total_tokens == 0:
        return {"tokens": 0, "avg_nll": float("nan"), "perplexity": float("nan")}
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return {"tokens": total_tokens, "avg_nll": avg_nll, "perplexity": ppl}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dirs = [s.strip() for s in args.data_dirs.split(",") if s.strip()]
    print("Loading dataset(s)...")
    raw = load_concat_datasets(data_dirs)
    if args.shuffle:
        raw = raw.shuffle(seed=args.seed)
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))
    print(f"Rows (pre-tokenization): {len(raw):,}")

    tok_src = args.tokenizer_path_or_id or args.model_path_or_id
    print(f"Loading tokenizer from: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    print("Packing dataset into fixed-length blocks...")
    ds = tokenize_and_group(tokenizer, raw, args.seq_len)
    print(f"Blocks for eval: {len(ds):,} (block size={args.seq_len})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = DataLoader(  # type: ignore[arg-type]
        ds,  # type: ignore
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
    )

    # Decide evaluation targets: single model or all checkpoints
    targets = _list_checkpoints(args.model_path_or_id)
    model_name = _model_name_from_path_or_id(args.model_path_or_id)

    # Optional Weights & Biases logging
    report_to = os.environ.get("EVAL_REPORT_TO", "none").lower()
    use_wandb = report_to == "wandb"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore

            wandb = _wandb
            proj = os.environ.get("WANDB_PROJECT", "EvalLM")
            run_name = f"eval-ppl-{model_name}"
            wandb.init(
                project=proj,
                name=run_name,
                config={
                    "model": args.model_path_or_id,
                    "data_dirs": data_dirs,
                    "seq_len": args.seq_len,
                    "batch_size": args.per_device_eval_batch_size,
                },
            )
        except Exception as e:
            print(f"[WARN] wandb not available or failed to init: {e}")
            use_wandb = False

    # Prepare output directory for JSON results
    out_root = os.path.join("eval", model_name)
    os.makedirs(out_root, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for item in tqdm(targets, desc="Evaluating checkpoints"):
        ckpt_path = cast(str, item["path"])  # type: ignore[index]
        step = item["step"]
        label = "final" if step is None else f"step-{step}"
        print(f"\n[Eval] {label}: {ckpt_path}")
        metrics = compute_ppl_for_model(ckpt_path, dl, device)
        result = {
            "checkpoint": ckpt_path,
            "label": label,
            "step": step,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seq_len": args.seq_len,
            "batch_size": args.per_device_eval_batch_size,
            **metrics,
        }
        # Save per-checkpoint result
        out_file = os.path.join(out_root, f"{label}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        summary.append(result)
        # Log to wandb (single run with step = global step or 0 for final)
        if use_wandb and wandb is not None:
            step_val = 0 if step is None else int(step)
            try:
                wandb.log(
                    {
                        "ppl": metrics["perplexity"],
                        "avg_nll": metrics["avg_nll"],
                        "tokens": metrics["tokens"],
                    },
                    step=step_val,
                )
            except Exception as e:
                print(f"[WARN] wandb.log failed: {e}")

        # Free GPU memory between checkpoints
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved eval results to: {out_root}")

    if use_wandb and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
