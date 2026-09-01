#!/usr/bin/env python3
"""Build a Hugging Face style dataset from batch completion outputs."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import Dataset, DatasetDict


def parse_record(raw: Dict) -> Optional[Dict[str, str]]:
    """Extract a single dataset row from a batch response entry."""

    response = raw.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None or not content.strip():
        return None

    record_id = raw.get("custom_id") or body.get("id") or raw.get("id")
    if not record_id:
        return None

    return {"id": str(record_id), "indonesian": content}


def iter_records(results_dir: Path) -> Iterable[Dict[str, str]]:
    """Yield parsed records from every JSONL file under results_dir."""

    jsonl_paths = sorted(results_dir.glob("*.jsonl"))
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Failed to parse JSON in {path} at line {line_number}"
                    )
                record = parse_record(raw)
                if record:
                    yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("batches/results"),
        help="Directory containing batch output JSONL files.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/synthetic_gen"),
        help="Destination directory for the Hugging Face dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to use for the generated dataset JSONL file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite dataset directory if it already exists.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    records = list(iter_records(args.results_dir))
    deduped = {}
    for record in records:
        deduped[record["id"]] = record

    dataset_dir = args.dataset_dir
    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset = Dataset.from_list(list(deduped.values()))
    dataset_dict = DatasetDict({args.split: hf_dataset})
    dataset_dict.save_to_disk(str(dataset_dir))

    print(f"Saved {len(hf_dataset)} rows to Hugging Face dataset at {dataset_dir}")


if __name__ == "__main__":
    main()
