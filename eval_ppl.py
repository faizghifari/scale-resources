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
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride (< seq_len enables sliding-window PPL). Default: no sliding (use non-overlapping blocks).",
    )
    p.add_argument(
        "--drop_first_window",
        action="store_true",
        help="When using sliding-window PPL, don't score tokens from the first window of each document.",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--shuffle", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of rows before tokenization (for a quick check).",
    )
    p.add_argument(
        "--report_to",
        type=str,
        choices=["none", "wandb"],
        default="wandb",
        help="Where to report metrics. Defaults to 'wandb'.",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="EvalLM",
        help="Weights & Biases project name (when --report_to=wandb).",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name. Defaults to 'eval-ppl-<model>'.",
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


def build_token_stream(tokenizer: Any, ds: Dataset) -> List[int]:
    """Tokenize dataset into a single token stream with EOS between docs."""

    def tok(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

    tokenized = ds.map(tok, batched=True, remove_columns=[], desc="Tokenizing (stream)")
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
    stream: List[int] = []
    for ids in tokenized["input_ids"]:
        if not ids:
            continue
        stream.extend(ids)
        stream.append(eos_id)
    return stream


def build_token_docs(tokenizer: Any, ds: Dataset) -> List[List[int]]:
    """Tokenize dataset into a list of token lists, one per document (no EOS joins)."""

    def tok(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

    tokenized = ds.map(tok, batched=True, desc="Tokenizing (per-doc)")
    input_ids_col = tokenized["input_ids"]
    # Ensure it's a list of lists of ints
    docs: List[List[int]] = []
    for ids in input_ids_col:
        if ids and isinstance(ids, list):
            docs.append([int(t) for t in ids])
        else:
            docs.append([])
    return docs


def compute_ppl_for_model_sliding(
    model_path: str,
    docs_token_ids: List[List[int]],
    tokenizer: Any,
    seq_len: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    drop_first_window: bool = False,
) -> Dict[str, Any]:
    """Compute per-document sliding-window perplexity.

    - No EOS between documents.
    - Score only the last `stride` tokens of each window (or up to L-1 if shorter).
    - If drop_first_window=True, the first window per document is not scored (n_pred=0).
    """
    assert stride > 0 and seq_len > 1 and stride <= seq_len
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for padding.")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = cast(torch.nn.Module, model).to(device)  # type: ignore[call-arg]
    model.eval()

    # Preflight: validate token ids fit model vocab to avoid device-side asserts
    try:
        embed = cast(Any, model).get_input_embeddings()
        vocab_size = int(embed.num_embeddings)  # type: ignore[attr-defined]
    except Exception:
        vocab_size = None  # type: ignore[assignment]
    if vocab_size is not None:
        # max/min over docs (fast) rather than scanning every window later
        max_id = -1
        min_id = 0
        for doc in docs_token_ids:
            if not doc:
                continue
            dmax = max(doc)
            dmin = min(doc)
            if dmax > max_id:
                max_id = dmax
            if dmin < min_id:
                min_id = dmin
        if max_id >= vocab_size or min_id < 0:
            raise ValueError(
                f"Token id out of range for model vocab: min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}.\n"
                f"This usually indicates a tokenizer/model mismatch. Ensure --tokenizer_path_or_id matches the model."
            )
        # Ensure pad_id is valid; if not, fall back to eos or 0
        if pad_id is None or pad_id < 0 or pad_id >= vocab_size:
            fallback = (
                eos_id if (eos_id is not None and 0 <= eos_id < vocab_size) else 0
            )
            print(
                f"[WARN] pad_token_id {pad_id} invalid for vocab_size {vocab_size}. Falling back to {fallback}."
            )
            pad_id = int(fallback)

    total_tokens = 0
    total_nll = 0.0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype: Optional[torch.dtype] = torch.float16 if device_type == "cuda" else None

    def make_batch(batch_windows: List[List[int]], batch_npred: List[int]):
        input_batch: List[List[int]] = []
        label_batch: List[List[int]] = []
        attn_batch: List[List[int]] = []
        for win, n_pred in zip(batch_windows, batch_npred):
            L = len(win)
            n_pred = min(n_pred, max(0, L - 1))
            prefix = L - n_pred
            labels = ([-100] * prefix) + win[prefix:]
            attn = [1] * L
            # pad to seq_len
            if L < seq_len:
                pad_len = seq_len - L
                win = win + [pad_id] * pad_len
                labels = labels + ([-100] * pad_len)
                attn = attn + ([0] * pad_len)
            input_batch.append(win)
            label_batch.append(labels)
            attn_batch.append(attn)
        return (
            torch.tensor(input_batch, dtype=torch.long),
            torch.tensor(label_batch, dtype=torch.long),
            torch.tensor(attn_batch, dtype=torch.long),
        )

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            batch_windows: List[List[int]] = []
            batch_npred: List[int] = []
            for doc_ids in docs_token_ids:
                if not doc_ids:
                    continue
                idx = 0
                first = True
                end_limit = len(doc_ids)
                while idx < end_limit - 1:
                    end = min(idx + seq_len, end_limit)
                    win = doc_ids[idx:end]
                    n_pred = (
                        0
                        if (first and drop_first_window)
                        else min(stride, max(0, len(win) - 1))
                    )
                    batch_windows.append(win)
                    batch_npred.append(n_pred)
                    if len(batch_windows) == batch_size:
                        tokens_this = sum(batch_npred)
                        if tokens_this > 0:
                            input_ids, labels, attn = make_batch(
                                batch_windows, batch_npred
                            )
                            input_ids = input_ids.to(device)
                            labels = labels.to(device)
                            attn = attn.to(device)
                            outputs = cast(Any, model)(
                                input_ids=input_ids, attention_mask=attn, labels=labels
                            )
                            loss = float(outputs.loss.item())
                            total_tokens += tokens_this
                            total_nll += loss * tokens_this
                        batch_windows = []
                        batch_npred = []
                    idx += stride
                    first = False
            # flush remaining
            if batch_windows:
                tokens_this = sum(batch_npred)
                if tokens_this > 0:
                    input_ids, labels, attn = make_batch(batch_windows, batch_npred)
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    attn = attn.to(device)
                    outputs = cast(Any, model)(
                        input_ids=input_ids, attention_mask=attn, labels=labels
                    )
                    loss = float(outputs.loss.item())
                    total_tokens += tokens_this
                    total_nll += loss * tokens_this

    if total_tokens == 0:
        return {"tokens": 0, "avg_nll": float("nan"), "perplexity": float("nan")}
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return {"tokens": total_tokens, "avg_nll": avg_nll, "perplexity": ppl}


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

    use_sliding = (
        args.stride is not None and args.stride > 0 and args.stride <= args.seq_len
    )
    if use_sliding:
        print(
            f"Tokenizing per-document for sliding-window eval (no EOS joins) (seq_len={args.seq_len}, stride={args.stride}, drop_first_window={args.drop_first_window})..."
        )
        token_docs = build_token_docs(tokenizer, raw)
        total_tokens_docs = sum(len(x) for x in token_docs)
        print(f"Docs: {len(token_docs):,}, total doc tokens: {total_tokens_docs:,}")
    else:
        print("Packing dataset into fixed-length blocks...")
        ds = tokenize_and_group(tokenizer, raw, args.seq_len)
        print(f"Blocks for eval: {len(ds):,} (block size={args.seq_len})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = None
    if not use_sliding:
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
    report_to = (
        args.report_to.lower()
        if args.report_to
        else os.environ.get("EVAL_REPORT_TO", "none").lower()
    )
    use_wandb = report_to == "wandb"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore

            wandb = _wandb
            proj = args.wandb_project or os.environ.get("WANDB_PROJECT", "EvalLM")
            run_name = args.wandb_run_name or model_name
            wandb.init(
                project=proj,
                name=run_name,
                config={
                    "model": args.model_path_or_id,
                    "data_dirs": data_dirs,
                    "seq_len": args.seq_len,
                    "batch_size": args.per_device_eval_batch_size,
                    "stride": args.stride,
                    "drop_first_window": args.drop_first_window,
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
        if use_sliding:
            metrics = compute_ppl_for_model_sliding(
                ckpt_path,
                token_docs,  # type: ignore[arg-type]
                tokenizer,
                args.seq_len,
                cast(int, args.stride),
                args.per_device_eval_batch_size,
                device,
                drop_first_window=args.drop_first_window,
            )
        else:
            assert dl is not None
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
