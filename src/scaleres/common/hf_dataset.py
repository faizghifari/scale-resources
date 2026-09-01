"""Shared Hugging Face ``datasets`` loading/normalization helpers.

Used by scaleres.training.train, train_tokenizer, and scaleres.eval.eval_ppl,
which otherwise each reimplemented an identical trio of
``_as_dataset``/``ensure_text_column``/``load_concat_datasets`` helpers.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

DEFAULT_TEXT_COLUMN_CANDIDATES: Sequence[str] = ("balinese", "cirebonese", "indonesian")


def as_dataset(obj: "Dataset | DatasetDict") -> Dataset:
    if isinstance(obj, DatasetDict):
        return concatenate_datasets([v for v in obj.values()])
    return obj


def ensure_text_column(
    ds: Dataset, candidates: Sequence[str] = DEFAULT_TEXT_COLUMN_CANDIDATES
) -> Dataset:
    """Normalize dataset to have a single 'text' column.

    Preference order:
    - If 'text' exists, keep it and drop other columns.
    - Else, if one of ``candidates`` exists, keep that and rename to 'text'.
    - Else, if only one column exists, rename it to 'text'.
    - Otherwise, raise an error.
    """
    cols = list(ds.column_names)
    if "text" in cols:
        remove_cols = [c for c in cols if c != "text"]
        return ds.remove_columns(remove_cols) if remove_cols else ds
    for c in candidates:
        if c in cols:
            tmp = (
                ds.remove_columns([x for x in cols if x != c]) if len(cols) > 1 else ds
            )
            return tmp.rename_column(c, "text") if c != "text" else tmp
    if len(cols) == 1:
        only = cols[0]
        return ds.rename_column(only, "text") if only != "text" else ds
    raise ValueError(f"No suitable text column in dataset with columns: {cols}")


def load_concat_datasets(
    dir_list: List[str],
    split_name: str = "dataset",
    candidates: Sequence[str] = DEFAULT_TEXT_COLUMN_CANDIDATES,
    warn_to_stderr: bool = False,
) -> Dataset:
    """Load, concatenate, and text-normalize one or more save_to_disk dataset dirs."""

    warn_stream = sys.stderr if warn_to_stderr else None
    datasets = []
    for d in dir_list:
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] {split_name} dataset dir not found: {d}", file=warn_stream)
            continue
        try:
            ds_any = load_from_disk(d)
            ds = as_dataset(ds_any)
            ds = ensure_text_column(ds, candidates)
            ds = ds.filter(
                lambda x: x["text"] is not None and str(x["text"]).strip() != ""
            )
            datasets.append(ds)
            print(f"Loaded {split_name} from {d}: {len(ds):,} rows")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Failed to load {d}: {e}", file=warn_stream)
    if not datasets:
        raise RuntimeError(f"No {split_name} datasets loaded from: {dir_list}")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def tokenize_and_group(
    tokenizer: Any, ds: Dataset, seq_len: int, compact_blocks: bool = False
) -> Dataset:
    """Tokenize and pack into fixed-length causal-LM blocks.

    ``compact_blocks`` stores ONLY input_ids, dropping attention_mask and labels.
    Both are redundant on disk and cost 2/3 of the cache:

      attention_mask  every block is exactly seq_len with no padding, so it is
                      all 1s, and tokenizer.pad() in the collator rebuilds it.
      labels          an exact copy of input_ids. DataCollatorForLanguageModeling
                      (mlm=False) clones input_ids into labels and masks pad
                      positions with -100 -- and there are no pad positions here,
                      so the result is identical to the copy stored on disk.

    Why it matters: three int64 lists per 512-token block is 12,288 bytes where
    1,024 would do. On a ~1B-token mixture that is a 12GB map cache the
    dataloader then reads at random, and once it exceeds page cache the GPU sits
    idle waiting on the disk. Measured on mix_full_real4x: 0.4 s/it early, 3.2
    s/it once the cache no longer fit -- an 8x slowdown with the GPU at 0%.

    Off by default so previously measured runs stay bit-reproducible. Callers
    that never pass ``compact_blocks=True`` (eval_ppl.py) get output identical
    to their pre-refactor local implementation.
    """

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
            "Tokenizer must define an eos_token_id to insert EOS between documents."
        )

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated_ids: List[int] = []
        if "input_ids" in examples:
            for ids in examples["input_ids"]:
                if not ids:
                    continue
                concatenated_ids.extend(ids)
                concatenated_ids.append(eos_id)
        total_length = (len(concatenated_ids) // seq_len) * seq_len
        if total_length == 0:
            return (
                {"input_ids": []}
                if compact_blocks
                else {"input_ids": [], "labels": [], "attention_mask": []}
            )
        input_blocks = [
            concatenated_ids[i : i + seq_len] for i in range(0, total_length, seq_len)
        ]
        if compact_blocks:
            return {"input_ids": input_blocks}
        attn_blocks = [[1] * seq_len for _ in range(len(input_blocks))]
        labels_blocks = [blk.copy() for blk in input_blocks]
        return {
            "input_ids": input_blocks,
            "attention_mask": attn_blocks,
            "labels": labels_blocks,
        }

    # In compact mode group_texts returns ONLY input_ids, so the tokenizer's own
    # attention_mask column would survive the map at the pre-packing row count and
    # collide with the packed one ("expected length 182 but got length 1000").
    # Dropping the input columns leaves exactly what group_texts returned. The
    # legacy path overwrites all three columns itself, so it is left untouched to
    # keep previously measured runs bit-reproducible.
    group_kwargs = {"remove_columns": tokenized.column_names} if compact_blocks else {}
    lm_ds = tokenized.map(
        group_texts, batched=True, desc="Grouping into blocks", **group_kwargs
    )
    return lm_ds
