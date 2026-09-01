"""Shared helpers for resumable per-topic JSONL pipelines (custom_id bookkeeping).

Used by scaleres.generation.generate_synthetic_answers and translate_answers,
which otherwise each reimplemented the same skip-id / append / path logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

JSONL_SUFFIX = ".jsonl"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_custom_ids_from_jsonl(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id")
            if isinstance(custom_id, str):
                ids.add(custom_id)
    return ids


def load_existing_output_ids(output_dir: Path) -> Set[str]:
    if not output_dir.exists():
        return set()
    ids: Set[str] = set()
    for path in sorted(output_dir.glob(f"*{JSONL_SUFFIX}")):
        ids.update(read_custom_ids_from_jsonl(path))
    return ids


def load_aggregate_ids(aggregate_path: Optional[Path]) -> Set[str]:
    if not aggregate_path:
        return set()
    return read_custom_ids_from_jsonl(aggregate_path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def topic_output_path(output_dir: Path, topic_index: int) -> Path:
    return output_dir / f"topic{topic_index:04d}{JSONL_SUFFIX}"


def load_existing_topic_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return read_custom_ids_from_jsonl(path)
