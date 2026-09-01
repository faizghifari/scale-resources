#!/usr/bin/env python
"""Run ``translate_answers.py`` in manageable topic batches.

This helper prevents loading every topic JSONL file at once by invoking the
translator repeatedly with a bounded ``--topic-limit``. A typical call processes
100 topic shards per step::

    python -m scaleres.generation.run_translate_chunks --chunk-size 100 -- \
        --api-base http://localhost:8000/v1 --model gpt-oss-120b

All arguments after ``--`` are forwarded directly to ``translate_answers.py``.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

JSONL_SUFFIX = ".jsonl"
TOPIC_PATTERN = re.compile(r"topic(\d{4})" + re.escape(JSONL_SUFFIX) + "$")
DEFAULT_TRANSLATOR_MODULE = "scaleres.generation.translate_answers"


def discover_topic_indices(
    input_dir: Path,
    min_index: int,
    max_index: int | None,
) -> List[int]:
    indices: List[int] = []
    for path in sorted(input_dir.glob(f"*{JSONL_SUFFIX}")):
        match = TOPIC_PATTERN.fullmatch(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx < min_index:
            continue
        if max_index is not None and idx > max_index:
            continue
        indices.append(idx)
    return indices


def chunk_indices(indices: List[int], chunk_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def build_command(
    translator_module: str,
    input_dir: Path,
    output_dir: Path,
    chunk: List[int],
    extra_args: List[str],
) -> List[str]:
    if not chunk:
        raise ValueError("Chunk must contain at least one topic index")
    offset = chunk[0]
    limit = len(chunk)
    cmd = [
        sys.executable,
        "-m",
        translator_module,
        "translate",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--topic-offset",
        str(offset),
        "--topic-limit",
        str(limit),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_chunks(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    chunk_size = args.chunk_size

    if chunk_size <= 0:
        raise SystemExit("--chunk-size must be a positive integer")
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.translate_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    indices = discover_topic_indices(input_dir, args.min_index, args.max_index)
    if not indices:
        print("No topic shards found that match the requested range.")
        return

    total_chunks = math.ceil(len(indices) / chunk_size)
    print(
        f"Discovered {len(indices)} topic shards in {input_dir}. "
        f"Processing in {total_chunks} chunk(s) of up to {chunk_size}."
    )

    for chunk_idx, chunk in enumerate(chunk_indices(indices, chunk_size), start=1):
        cmd = build_command(
            args.translator_module, input_dir, output_dir, chunk, extra_args
        )
        start_idx = chunk[0]
        end_idx = chunk[-1]
        print(
            f"[{chunk_idx}/{total_chunks}] Translating topics index range {start_idx:04d}-{end_idx:04d}"
        )
        if args.dry_run:
            print("  Dry run: ", " ".join(cmd))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:  # noqa: BLE001
            print(f"  Chunk failed with exit code {exc.returncode}.")
            if not args.continue_on_error:
                raise
            print("  Continuing to next chunk because --continue-on-error is set.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute translate_answers.py in bounded topic batches.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--translator-module",
        type=str,
        default=DEFAULT_TRANSLATOR_MODULE,
        help="Module to invoke via `python -m` (default: scaleres.generation.translate_answers).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("synthetic_data/raw/answers"),
        help="Directory containing per-topic answer JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_data/raw/translations"),
        help="Directory where translation outputs should be written.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Maximum number of topic files to translate per invocation.",
    )
    parser.add_argument(
        "--min-index",
        type=int,
        default=0,
        help="Ignore topic files whose index is lower than this value.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        help="Upper bound on topic file indices to include (inclusive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip to the next chunk if a subprocess exits with a failure code.",
    )
    parser.add_argument(
        "translate_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to translate_answers.py."
        " Prefix with -- to separate.",
    )
    args = parser.parse_args()
    if args.max_index is not None and args.max_index < args.min_index:
        parser.error("--max-index must be greater than or equal to --min-index")
    return args


def main() -> None:
    args = parse_args()
    run_chunks(args)


if __name__ == "__main__":
    main()
