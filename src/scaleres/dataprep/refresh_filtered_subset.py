#!/usr/bin/env python3
"""Recompute the filtered_heuristic split with the latest heuristic settings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from .quality_utils import (
    LANG_CONFIGS,
    GlotLanguageIdentifier,
    LangConfig,
    evaluate_text,
)

LANG_ORDER: Sequence[str] = ("balinese", "cirebonese")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the filtered_heuristic subset by re-running language heuristics "
            "over the raw split."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/parallel/synthetic"),
        help="Path to the existing DatasetDict containing 'raw' and 'filtered_heuristic' splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write the updated DatasetDict. Defaults to --dataset-dir.",
    )
    parser.add_argument(
        "--lid-model",
        type=Path,
        default=Path("models/glotlid/model.bin"),
        help="Path to the GlotLID FastText model used for language identification.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during evaluation.",
    )
    return parser.parse_args()


def evaluate_language(
    dataset: Dataset,
    field_name: str,
    lang_key: str,
    lid_model: Path,
    show_progress: bool,
) -> Tuple[List[bool], Counter]:
    config: LangConfig = LANG_CONFIGS[lang_key]
    lid = GlotLanguageIdentifier(lid_model, config)
    total = len(dataset)
    iterator = dataset
    if show_progress:
        iterator = tqdm(
            dataset,
            total=total,
            desc=f"heuristics:{lang_key}",
            unit="rows",
        )

    decisions: List[bool] = []
    rejects: Counter = Counter()
    for row in iterator:
        text = row.get(field_name) if isinstance(row, dict) else None
        text_str = text if isinstance(text, str) else ""
        eval_result = evaluate_text(text_str, config, lid)
        decisions.append(eval_result.ok)
        if not eval_result.ok:
            reason = eval_result.reason or "unknown"
            rejects[reason] += 1

    return decisions, rejects


def filter_dataset(
    raw: Dataset,
    lid_model: Path,
    show_progress: bool,
) -> Tuple[Dataset, Dict[str, object]]:
    masks: Dict[str, List[bool]] = {}
    reject_stats: Dict[str, Counter] = {}

    for lang_key in LANG_ORDER:
        field_name = lang_key
        mask, rejects = evaluate_language(
            raw,
            field_name=field_name,
            lang_key=lang_key,
            lid_model=lid_model,
            show_progress=show_progress,
        )
        masks[lang_key] = mask
        reject_stats[lang_key] = rejects

    keep_indices: List[int] = []
    for idx, flags in enumerate(zip(*(masks[key] for key in LANG_ORDER))):
        if all(flags):
            keep_indices.append(idx)

    filtered = raw.select(keep_indices)
    summary: Dict[str, object] = {
        "raw_rows": len(raw),
        "filtered_rows": len(filtered),
        "balinese_kept": sum(masks["balinese"]),
        "cirebonese_kept": sum(masks["cirebonese"]),
        "balinese_rejects": {k: int(v) for k, v in reject_stats["balinese"].items()},
        "cirebonese_rejects": {
            k: int(v) for k, v in reject_stats["cirebonese"].items()
        },
    }
    return filtered, summary


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_dir
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory {dataset_path} not found. Build the parallel dataset first."
        )

    ds = load_from_disk(str(dataset_path))
    if not isinstance(ds, DatasetDict) or "raw" not in ds:
        raise ValueError(
            "Expected a DatasetDict with at least the 'raw' split present at the dataset path."
        )

    raw = ds["raw"]
    raw.set_format("python")

    filtered, summary = filter_dataset(
        raw,
        lid_model=args.lid_model,
        show_progress=not args.no_progress,
    )
    raw.reset_format()

    output_dir = args.output_dir
    if output_dir:
        ds_out = DatasetDict({"raw": raw, "filtered_heuristic": filtered})
        ds_out.save_to_disk(str(output_dir))
        save_path = output_dir
    else:
        filtered.save_to_disk(str(dataset_path / "filtered_heuristic"))
        save_path = dataset_path

    print(f"Filtering summary (written to {save_path}):")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
