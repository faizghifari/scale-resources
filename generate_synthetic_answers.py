#!/usr/bin/env python
"""Generate synthetic answers using a vLLM deployment with OpenAI-compatible APIs.

This refactor removes the legacy OpenAI Batch workflow and replaces it with a
fully asynchronous pipeline that talks directly to a vLLM server. The script
uses two layers of concurrency (topic-level and request-level) mirroring the
``generate_topics.py`` design, skips questions that were already answered in
previous batch runs, and emits JSONL outputs per topic that remain compatible
with ``create_synthetic_dataset.py``.

Typical usages::

    # Generate answers for all remaining questions and store per-topic JSONL
    python generate_synthetic_answers.py generate \
        --input generated_topics.json \
        --output-dir topic_answers \
        --request-concurrency 64 \
        --topic-concurrency 4 \
        --api-base http://localhost:8000/v1 \
        --model gpt-oss-120b

    # Run a quick concurrent sample without writing outputs
    python generate_synthetic_answers.py sample --count 3

The script automatically loads existing batch request JSONL files listed in
``batches/.submitted_batches`` (and any legacy results) to avoid re-processing
questions that were already handled. Existing per-topic output files produced by
this script are also respected, enabling resumable executions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant who always responds in fluent, natural Indonesian. "
    "Deliver only the final answer; never describe internal reasoning or thought processes. "
    "Reduce the use of bullet points, numbered lists, headings, or any list-like formatting; only use it when really necessary. "
    "Vary sentence length and tone to keep the prose engaging, weaving in relevant examples, comparisons, or brief illustrative anecdotes when helpful. "
    "Do not repeat or rephrase the question; begin directly with the answer in Indonesian language."
)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "token-abc123"
DEFAULT_VLLM_MODEL = "gpt-oss-120b"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TIMEOUT = 180.0
DEFAULT_TOPIC_CONCURRENCY = 1
DEFAULT_REQUEST_CONCURRENCY = 512
DEFAULT_MAX_RETRIES = 2
DEFAULT_REQUEST_DELAY = 0.0

JSONL_SUFFIX = ".jsonl"


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    request_delay: float = DEFAULT_REQUEST_DELAY
    model: str = DEFAULT_VLLM_MODEL
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuestionItem:
    topic_index: int
    subtopic_index: int
    question_index: int
    custom_id: str
    topic: str
    subtopic: str
    question: str


@dataclass(slots=True)
class TopicBundle:
    index: int
    title: str
    questions: List[QuestionItem]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_topics(input_path: Path) -> List[TopicBundle]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    bundles: List[TopicBundle] = []
    for topic_idx, topic_entry in enumerate(data):
        topic_title = str(topic_entry.get("topic", "")).strip()
        questions: List[QuestionItem] = []
        for sub_idx, sub_entry in enumerate(topic_entry.get("subtopics", [])):
            subtopic_title = str(sub_entry.get("title", "")).strip()
            for question_idx, question in enumerate(sub_entry.get("questions", [])):
                custom_id = f"topic{topic_idx:04d}-sub{sub_idx:03d}-q{question_idx:03d}"
                questions.append(
                    QuestionItem(
                        topic_index=topic_idx,
                        subtopic_index=sub_idx,
                        question_index=question_idx,
                        custom_id=custom_id,
                        topic=topic_title,
                        subtopic=subtopic_title,
                        question=str(question),
                    )
                )
        if questions:
            bundles.append(
                TopicBundle(index=topic_idx, title=topic_title, questions=questions)
            )
    return bundles


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


def load_submitted_custom_ids(batches_dir: Path, submitted_path: Path) -> Set[str]:
    if not submitted_path.exists() or not batches_dir.exists():
        return set()

    submitted_ids: Set[str] = set()
    entries = [
        line.strip()
        for line in submitted_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in entries:
        candidate_paths: List[Path]
        if entry.endswith(JSONL_SUFFIX):
            candidate = batches_dir / entry
            candidate_paths = [candidate]
        else:
            candidate_paths = sorted(batches_dir.glob(f"{entry}*{JSONL_SUFFIX}"))
        for candidate in candidate_paths:
            if candidate.exists():
                submitted_ids.update(read_custom_ids_from_jsonl(candidate))
                break
    return submitted_ids


def load_legacy_result_ids(results_dir: Path) -> Set[str]:
    if not results_dir.exists():
        return set()
    ids: Set[str] = set()
    for path in sorted(results_dir.glob(f"*{JSONL_SUFFIX}")):
        ids.update(read_custom_ids_from_jsonl(path))
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


def parse_extra_body(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {"include_reasoning": False}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid JSON for --extra-body: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return payload


def build_client(args: argparse.Namespace) -> AsyncOpenAI:
    load_dotenv()
    api_base = args.api_base or DEFAULT_VLLM_BASE_URL
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or DEFAULT_VLLM_API_KEY
    return AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=args.timeout)


async def create_chat_completion(
    client: AsyncOpenAI,
    item: QuestionItem,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
) -> ChatCompletion:
    backoff_base = 2.0
    last_error: Exception | None = None

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": item.question},
    ]

    for attempt in range(1, cfg.max_retries + 1):
        try:
            async with request_semaphore:
                response = await client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,  # type: ignore
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                    extra_body=cfg.extra_body or None,
                )
            if cfg.request_delay:
                await asyncio.sleep(cfg.request_delay)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= cfg.max_retries:
                raise
            sleep_for = backoff_base ** (attempt - 1)
            await asyncio.sleep(sleep_for + random.uniform(0, 0.5))

    raise RuntimeError("Unreachable state in create_chat_completion") from last_error


def build_response_record(
    item: QuestionItem, response: ChatCompletion
) -> Dict[str, Any]:
    payload = response.model_dump()
    created_ts = payload.get("created")
    if not created_ts:
        created_ts = int(time.time())
        payload["created"] = created_ts

    return {
        "id": f"req_{item.custom_id}",
        "custom_id": item.custom_id,
        "topic": item.topic,
        "subtopic": item.subtopic,
        "question": item.question,
        "response": {
            "status_code": 200,
            "request_id": None,
            "body": payload,
        },
        "error": None,
    }


async def fetch_answer_record(
    client: AsyncOpenAI,
    item: QuestionItem,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    completion = await create_chat_completion(client, item, cfg, request_semaphore)
    return build_response_record(item, completion)


def topic_output_path(output_dir: Path, bundle: TopicBundle) -> Path:
    return output_dir / f"topic{bundle.index:04d}{JSONL_SUFFIX}"


def load_existing_topic_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return read_custom_ids_from_jsonl(path)


async def process_topic(
    bundle: TopicBundle,
    client: AsyncOpenAI,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
    output_dir: Path,
    skip_ids: Set[str],
    skip_lock: asyncio.Lock,
    progress_bar: tqdm,
) -> Tuple[int, List[Tuple[str, str]]]:
    path = topic_output_path(output_dir, bundle)
    existing_ids = load_existing_topic_ids(path)

    pending: List[QuestionItem] = [
        item
        for item in bundle.questions
        if item.custom_id not in skip_ids and item.custom_id not in existing_ids
    ]

    if not pending:
        progress_bar.write(
            f"Topic {bundle.index:04d} skipped (all questions already processed)."
        )
        return 0, []

    ensure_directory(output_dir)
    topic_lock = asyncio.Lock()
    written = 0
    failures: List[Tuple[str, str]] = []

    async def worker(item: QuestionItem) -> None:
        nonlocal written
        try:
            record = await fetch_answer_record(client, item, cfg, request_semaphore)
        except Exception as exc:  # noqa: BLE001
            failures.append((item.custom_id, str(exc)))
            return

        async with topic_lock:
            append_jsonl(path, record)
            written += 1
        async with skip_lock:
            skip_ids.add(item.custom_id)

    await asyncio.gather(*(worker(item) for item in pending))

    if failures:
        summary = "; ".join(f"{cid}: {msg}" for cid, msg in failures[:3])
        progress_bar.write(
            f"Topic {bundle.index:04d} finished with {written} answers and {len(failures)} failures: {summary}"
        )
    else:
        progress_bar.write(f"Topic {bundle.index:04d} finished with {written} answers.")

    return written, failures


def gather_skip_ids(args: argparse.Namespace, output_dir: Path) -> Set[str]:
    skip_ids: Set[str] = set()
    skip_ids.update(
        load_submitted_custom_ids(args.legacy_batches_dir, args.submitted_batches)
    )
    skip_ids.update(load_legacy_result_ids(args.legacy_results_dir))
    skip_ids.update(load_existing_output_ids(output_dir))
    skip_ids.update(load_aggregate_ids(args.aggregate_answers))
    return skip_ids


async def run_generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    ensure_directory(output_dir)

    bundles = load_topics(args.input)
    if not bundles:
        print("No topics found in input file; nothing to do.")
        return

    cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        model=args.model,
        extra_body=args.extra_body_payload,
    )

    skip_ids = gather_skip_ids(args, output_dir)
    skip_lock = asyncio.Lock()

    client = build_client(args)
    request_semaphore = asyncio.Semaphore(max(1, args.request_concurrency))
    topic_semaphore = asyncio.Semaphore(max(1, args.topic_concurrency))

    total_written = 0
    total_failures: List[Tuple[str, str]] = []

    progress_bar = tqdm(total=len(bundles), desc="Topics", unit="topic")

    async def topic_worker(bundle: TopicBundle) -> None:
        nonlocal total_written
        async with topic_semaphore:
            written, failures = await process_topic(
                bundle,
                client,
                cfg,
                request_semaphore,
                output_dir,
                skip_ids,
                skip_lock,
                progress_bar,
            )
            total_written += written
            total_failures.extend(failures)
            progress_bar.update(1)

    await asyncio.gather(*(topic_worker(bundle) for bundle in bundles))
    progress_bar.close()

    await client.close()

    print(f"Completed generation with {total_written} new answers.")
    if total_failures:
        print(
            f"Encountered {len(total_failures)} failures. See logs above for details."
        )


async def run_sample(args: argparse.Namespace) -> None:
    bundles = load_topics(args.input)
    if not bundles:
        print("No topics found in input file; nothing to sample.")
        return

    skip_ids = gather_skip_ids(args, args.output_dir)
    cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        model=args.model,
        extra_body=args.extra_body_payload,
    )

    pending: List[QuestionItem] = []
    for bundle in bundles:
        for item in bundle.questions:
            if item.custom_id not in skip_ids:
                pending.append(item)
            if len(pending) >= args.count:
                break
        if len(pending) >= args.count:
            break

    if not pending:
        print("No unanswered questions available for sampling.")
        return

    client = build_client(args)
    request_semaphore = asyncio.Semaphore(max(1, args.request_concurrency))

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    async def worker(item: QuestionItem) -> None:
        try:
            record = await fetch_answer_record(client, item, cfg, request_semaphore)
            results.append(record)
        except Exception as exc:  # noqa: BLE001
            failures.append((item.custom_id, str(exc)))

    await asyncio.gather(*(worker(item) for item in pending))
    await client.close()

    if args.sample_output:
        ensure_directory(args.sample_output.parent)
        for record in results:
            append_jsonl(args.sample_output, record)

    for record in results:
        answer = (
            record.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        preview = answer.replace("\n", " ")[:120]
        print(f"Sample {record['custom_id']}: {preview}...")

    if failures:
        print(f"Sample completed with {len(failures)} failures:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")


def parse_args() -> argparse.Namespace:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--input",
        type=Path,
        default=Path("generated_topics.json"),
        help="Path to the JSON file containing topics, subtopics, and questions.",
    )
    parent.add_argument(
        "--output-dir",
        type=Path,
        default=Path("topic_answers"),
        help="Directory to store per-topic answer JSONL files.",
    )
    parent.add_argument(
        "--api-base",
        default=DEFAULT_VLLM_BASE_URL,
        help="Base URL for the OpenAI-compatible vLLM server.",
    )
    parent.add_argument(
        "--api-key",
        help="API key/token for the vLLM server (defaults to VLLM_API_KEY or built-in token).",
    )
    parent.add_argument(
        "--model",
        default=DEFAULT_VLLM_MODEL,
        help="Model identifier to request from the vLLM backend.",
    )
    parent.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens for each answer.",
    )
    parent.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for generation.",
    )
    parent.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p nucleus sampling probability mass.",
    )
    parent.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout (seconds) for API requests.",
    )
    parent.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum number of retries for each failed request.",
    )
    parent.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Seconds to sleep after each successful request (for rate limiting).",
    )
    parent.add_argument(
        "--topic-concurrency",
        type=int,
        default=DEFAULT_TOPIC_CONCURRENCY,
        help="Maximum number of topics processed concurrently.",
    )
    parent.add_argument(
        "--request-concurrency",
        type=int,
        default=DEFAULT_REQUEST_CONCURRENCY,
        help="Maximum number of in-flight chat completion requests.",
    )
    parent.add_argument(
        "--submitted-batches",
        type=Path,
        default=Path("batches/.submitted_batches"),
        help="File listing batch identifiers that have already been processed.",
    )
    parent.add_argument(
        "--legacy-batches-dir",
        type=Path,
        default=Path("batches"),
        help="Directory containing legacy batch request JSONL files.",
    )
    parent.add_argument(
        "--legacy-results-dir",
        type=Path,
        default=Path("batches/results"),
        help="Directory containing legacy batch output JSONL files.",
    )
    parent.add_argument(
        "--aggregate-answers",
        type=Path,
        help="Optional aggregated JSONL file whose entries should be skipped.",
    )
    parent.add_argument(
        "--extra-body",
        help="Optional JSON string merged into the OpenAI extra_body payload.",
    )

    parser = argparse.ArgumentParser(description="Generate synthetic answers via vLLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        parents=[parent],
        help="Generate answers for all pending questions.",
    )
    generate_parser.set_defaults(func=lambda args: asyncio.run(run_generate(args)))

    sample_parser = subparsers.add_parser(
        "sample",
        parents=[parent],
        help="Run a concurrent sample without writing permanent outputs.",
    )
    sample_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of unanswered questions to fetch during the sample run.",
    )
    sample_parser.add_argument(
        "--sample-output",
        type=Path,
        help="Optional JSONL path to store sample responses for inspection.",
    )
    sample_parser.set_defaults(func=lambda args: asyncio.run(run_sample(args)))

    args = parser.parse_args()
    args.extra_body_payload = parse_extra_body(args.extra_body)
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
