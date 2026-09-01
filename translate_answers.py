#!/usr/bin/env python
"""Translate previously generated Indonesian answers into Balinese and Cirebonese via vLLM.

The script mirrors the asynchronous batching pattern used in ``generate_synthetic_answers.py``
but consumes the JSONL answer files in ``topic_answers/`` (or a user supplied directory).
For every answer it builds a prompt using ``instruction.txt`` plus automatic lexicon
lookups from ``dict/idn_bali.json`` and ``dict/idn_cbn.json``, then calls an OpenAI-compatible
vLLM endpoint concurrently. Each response must be a JSON object with ``balinese`` and
``cirebonese`` strings; outputs are written per-topic as JSONL files so runs are resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tiktoken
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise translation assistant. Follow every user instruction carefully. "
    "Never describe internal reasoning or thought processes. "
    "Always reply only with a single valid JSON object containing exactly two string keys: "
    '"balinese" and "cirebonese". Never add commentary, code fences, or extra text.'
)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "token-abc123"
DEFAULT_VLLM_MODEL = "gpt-oss-120b"
MODEL_CONTEXT_LIMIT = 16384
MIN_COMPLETION_TOKENS = 2048
DEFAULT_MAX_TOKENS = MODEL_CONTEXT_LIMIT - 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TIMEOUT = 360.0
DEFAULT_TOPIC_CONCURRENCY = 1
DEFAULT_REQUEST_CONCURRENCY = 128
DEFAULT_MAX_RETRIES = 2
DEFAULT_REQUEST_DELAY = 0.0

JSONL_SUFFIX = ".jsonl"
CODE_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n|```$", re.MULTILINE)
TOKEN_STRIP = "\u200b\ufeff"  # remove stray zero-width chars from inputs
TOPIC_FILE_PATTERN = re.compile(r"topic(\d{4})" + re.escape(JSONL_SUFFIX) + "$")
SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


@dataclass(slots=True)
class TranslationConfig:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = DEFAULT_MAX_RETRIES
    request_delay: float = DEFAULT_REQUEST_DELAY
    model: str = DEFAULT_VLLM_MODEL
    extra_body: Dict[str, Any] = field(default_factory=dict)
    context_limit: int = MODEL_CONTEXT_LIMIT


@dataclass(slots=True)
class AnswerItem:
    topic_index: int
    custom_id: str
    topic: str
    subtopic: str
    question: str
    answer: str


@dataclass(slots=True)
class TopicAnswerBundle:
    index: int
    path: Path
    items: List[AnswerItem]


@dataclass(slots=True)
class InstructionBlocks:
    instruction: str


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


def extract_topic_index(path: Path) -> Optional[int]:
    match = TOPIC_FILE_PATTERN.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_topic_paths(
    input_dir: Path,
    topic_offset: int = 0,
    topic_limit: Optional[int] = None,
) -> Iterable[Tuple[int, Path]]:
    processed = 0
    for path in sorted(input_dir.glob(f"*{JSONL_SUFFIX}")):
        index = extract_topic_index(path)
        if index is None:
            continue
        if index < topic_offset:
            continue
        if topic_limit is not None and processed >= topic_limit:
            break
        yield index, path
        processed += 1


def read_topic_items(path: Path, topic_index: int) -> List[AnswerItem]:
    items: List[AnswerItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("error"):
                continue
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str):
                continue

            topic = record.get("topic", "")
            subtopic = record.get("subtopic", "")
            question = record.get("question", "")

            answer = (
                record.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not isinstance(answer, str) or not answer.strip():
                continue

            items.append(
                AnswerItem(
                    topic_index=topic_index,
                    custom_id=custom_id,
                    topic=str(topic),
                    subtopic=str(subtopic),
                    question=str(question),
                    answer=answer,
                )
            )
    return items


@lru_cache(maxsize=16)
def load_chat_encoder(
    model: str, tokenizer_override: Optional[str] = None
) -> Callable[[str], List[int]]:
    if tokenizer_override:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # noqa: PYL-E0401
            raise SystemExit(
                "The 'transformers' package is required when --tokenizer-model is provided."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[attr-defined]
            tokenizer_override,
            trust_remote_code=True,
        )

        def encode_text(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False)  # type: ignore[no-any-return]

        return encode_text

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode


def count_prompt_tokens(
    system_prompt: str, user_prompt: str, encode_text: Callable[[str], List[int]]
) -> int:
    tokens_per_message = 3
    tokens_per_name = 1
    total_tokens = 0
    messages = (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    )
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            if key == "name":
                total_tokens += tokens_per_name
            total_tokens += len(encode_text(value))
    total_tokens += 3  # every reply is primed with <|assistant|>
    return total_tokens


def load_instruction_blocks(path: Path) -> InstructionBlocks:
    content = path.read_text(encoding="utf-8")
    instruction_text = extract_xml_block(content, "instruction")
    if instruction_text is None:
        raise SystemExit(
            "instruction.txt must contain <instruction>...</instruction> block"
        )
    return InstructionBlocks(instruction=instruction_text.strip())


def extract_xml_block(content: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(1)
    return None


def load_dictionary(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    # ensure keys are lower-case for matching
    return {str(k).lower(): [str(vv) for vv in value] for k, value in data.items()}


def strip_zero_width(text: str) -> str:
    return text.translate({ord(ch): None for ch in TOKEN_STRIP})


def tokenize_text(text: str) -> List[str]:
    raw_tokens = text.lower().split()
    tokens: List[str] = []
    for token in raw_tokens:
        cleaned = token.strip("\"'()[]{}.,;:!?<>«»“”‘’—-")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    for idx in range(len(tokens) - n + 1):
        yield " ".join(tokens[idx : idx + n])


def build_lexicon_block(text: str, dictionary: Dict[str, List[str]]) -> str:
    tokens = tokenize_text(text)
    seen: Set[str] = set()
    lines: List[str] = []
    for n in (3, 2, 1):
        for ngram in iter_ngrams(tokens, n):
            if ngram in seen:
                continue
            translations = dictionary.get(ngram)
            if translations:
                seen.add(ngram)
                joined = ", ".join(translations)
                lines.append(f"- {ngram}: {joined}")
    return "\n".join(lines) if lines else "None"


def sanitize_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        return SURROGATE_PATTERN.sub("\ufffd", value)
    if isinstance(value, list):
        return [sanitize_surrogates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_surrogates(item) for item in value)
    if isinstance(value, dict):
        return {
            (
                sanitize_surrogates(key) if isinstance(key, str) else key
            ): sanitize_surrogates(val)
            for key, val in value.items()
        }
    return value


def render_prompt(
    instruction: InstructionBlocks,
    text: str,
    balinese_lexicon: str,
    cirebonese_lexicon: str,
) -> str:
    safe_text = strip_zero_width(text).strip()
    parts = [
        "<instruction>",
        instruction.instruction,
        "</instruction>",
        "",
        "<text>",
        safe_text,
        "</text>",
        "",
        "<balinese_lexicon>",
        balinese_lexicon if balinese_lexicon.strip() else "None",
        "</balinese_lexicon>",
        "",
        "<cirebonese_lexicon>",
        cirebonese_lexicon if cirebonese_lexicon.strip() else "None",
        "</cirebonese_lexicon>",
        'Output exactly one minified JSON object with keys "balinese" and "cirebonese" and their string values; no other text before or after. No newlines or spaces outside of strings. Escape any double quotes inside values.',
    ]
    return "\n".join(parts)


def parse_extra_body(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
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
    return AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=args.timeout,
        default_headers={"OpenAI-Model": args.model},
    )


async def create_chat_completion(
    client: AsyncOpenAI,
    prompt: str,
    cfg: TranslationConfig,
    request_semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> ChatCompletion:
    backoff_base = 2.0
    last_error: Exception | None = None

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, cfg.max_retries + 1):
        try:
            async with request_semaphore:
                response = await client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,  # type: ignore
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=max_tokens,
                    reasoning_effort="low",
                    frequency_penalty=1.0,
                    response_format={"type": "json_object"},
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

    raise RuntimeError(
        "Reached unreachable state in create_chat_completion"
    ) from last_error


def clean_response_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = CODE_FENCE_PATTERN.sub("", stripped)
    return stripped.strip()


def parse_translation_json(raw: str) -> Dict[str, str]:
    cleaned = clean_response_content(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"Model response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object")

    missing = {"balinese", "cirebonese"} - set(payload)
    if missing:
        raise ValueError(f"Model response missing keys: {', '.join(sorted(missing))}")

    balinese = payload.get("balinese")
    cirebonese = payload.get("cirebonese")
    if not isinstance(balinese, str) or not isinstance(cirebonese, str):
        raise ValueError("Both 'balinese' and 'cirebonese' must be strings")

    return {"balinese": balinese, "cirebonese": cirebonese}


def build_translation_record(
    item: AnswerItem,
    prompt: str,
    balinese_lexicon: str,
    cirebonese_lexicon: str,
    completion: ChatCompletion,
    parsed: Optional[Dict[str, str]] = None,
    prompt_tokens: Optional[int] = None,
    planned_max_tokens: Optional[int] = None,
    context_limit: Optional[int] = None,
    raw_response: Optional[str] = None,
    parse_error: Optional[str] = None,
) -> Dict[str, Any]:
    payload = completion.model_dump()
    created_ts = payload.get("created")
    if not created_ts:
        created_ts = int(time.time())
        payload["created"] = created_ts

    record: Dict[str, Any] = {
        "id": f"trans_{item.custom_id}",
        "custom_id": item.custom_id,
        "topic": item.topic,
        "subtopic": item.subtopic,
        "question": item.question,
        "answer": item.answer,
        "prompt_input": {
            "text": item.answer,
            "balinese_lexicon": balinese_lexicon,
            "cirebonese_lexicon": cirebonese_lexicon,
        },
        "response": {
            "status_code": 200,
            "request_id": None,
            "body": payload,
        },
        "translations": parsed,
    }

    token_plan: Dict[str, int] = {}
    if prompt_tokens is not None:
        token_plan["prompt_tokens"] = prompt_tokens
    if planned_max_tokens is not None:
        token_plan["max_tokens"] = planned_max_tokens
    if context_limit is not None:
        token_plan["context_limit"] = context_limit
    if token_plan:
        record["token_plan"] = token_plan

    if raw_response is not None:
        record["raw_response"] = raw_response
    if parse_error is not None:
        record["parse_error"] = parse_error

    return record


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    sanitized = sanitize_surrogates(payload)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitized, ensure_ascii=False) + "\n")


def topic_output_path(output_dir: Path, topic_index: int) -> Path:
    return output_dir / f"topic{topic_index:04d}{JSONL_SUFFIX}"


def load_existing_topic_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return read_custom_ids_from_jsonl(path)


def gather_skip_ids(output_dir: Path, aggregate_path: Optional[Path]) -> Set[str]:
    skip_ids: Set[str] = set()
    skip_ids.update(load_existing_output_ids(output_dir))
    skip_ids.update(load_aggregate_ids(aggregate_path))
    return skip_ids


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


def load_answer_bundles(
    input_dir: Path,
    topic_offset: int = 0,
    topic_limit: Optional[int] = None,
) -> List[TopicAnswerBundle]:
    bundles: List[TopicAnswerBundle] = []
    for index, path in iter_topic_paths(
        input_dir, topic_offset=topic_offset, topic_limit=topic_limit
    ):
        items = read_topic_items(path, index)
        if items:
            bundles.append(TopicAnswerBundle(index=index, path=path, items=items))
    return bundles


async def process_topic(
    bundle: TopicAnswerBundle,
    client: AsyncOpenAI,
    cfg: TranslationConfig,
    request_semaphore: asyncio.Semaphore,
    output_dir: Path,
    skip_ids: Set[str],
    skip_lock: asyncio.Lock,
    progress_bar: tqdm,
    instruction: InstructionBlocks,
    balinese_dict: Dict[str, List[str]],
    cirebonese_dict: Dict[str, List[str]],
    tokenize: Callable[[str], List[int]],
) -> Tuple[int, List[Tuple[str, str]]]:
    path = topic_output_path(output_dir, bundle.index)
    existing_ids = load_existing_topic_ids(path)

    pending: List[AnswerItem] = [
        item
        for item in bundle.items
        if item.custom_id not in skip_ids and item.custom_id not in existing_ids
    ]

    if not pending:
        progress_bar.write(
            f"Topic {bundle.index:04d} skipped (all answers already translated)."
        )
        return 0, []

    ensure_directory(output_dir)
    topic_lock = asyncio.Lock()
    written = 0
    failures: List[Tuple[str, str]] = []

    async def worker(item: AnswerItem) -> None:
        nonlocal written
        balinese_lexicon = build_lexicon_block(item.answer, balinese_dict)
        cirebonese_lexicon = build_lexicon_block(item.answer, cirebonese_dict)
        prompt = render_prompt(
            instruction, item.answer, balinese_lexicon, cirebonese_lexicon
        )

        prompt_tokens = (
            count_prompt_tokens(DEFAULT_SYSTEM_PROMPT, prompt, tokenize) + 100
        )
        available_tokens = cfg.context_limit - prompt_tokens
        if available_tokens < MIN_COMPLETION_TOKENS:
            failures.append(
                (
                    item.custom_id,
                    f"Insufficient token budget: prompt={prompt_tokens}, limit={cfg.context_limit}",
                )
            )
            return
        max_tokens = min(cfg.max_tokens, available_tokens)

        try:
            completion = await create_chat_completion(
                client,
                prompt,
                cfg,
                request_semaphore,
                max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((item.custom_id, str(exc)))
            return

        message = completion.choices[0].message
        raw_content = message.content if message and message.content else ""
        parsed: Optional[Dict[str, str]] = None
        parse_error: Optional[str] = None
        try:
            parsed = parse_translation_json(raw_content)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            failures.append((item.custom_id, f"Stored with parse error: {parse_error}"))

        record = build_translation_record(
            item,
            prompt,
            balinese_lexicon,
            cirebonese_lexicon,
            completion,
            parsed,
            prompt_tokens=prompt_tokens,
            planned_max_tokens=max_tokens,
            context_limit=cfg.context_limit,
            raw_response=raw_content if parse_error else None,
            parse_error=parse_error,
        )

        try:
            async with topic_lock:
                append_jsonl(path, record)
                written += 1
        except UnicodeEncodeError as exc:
            failures.append((item.custom_id, f"Write failed: {exc}"))
            return
        async with skip_lock:
            skip_ids.add(item.custom_id)

    await asyncio.gather(*(worker(item) for item in pending))

    if failures:
        summary = "; ".join(f"{cid}: {msg}" for cid, msg in failures[:3])
        progress_bar.write(
            f"Topic {bundle.index:04d} finished with {written} translations and {len(failures)} failures: {summary}"
        )
    else:
        progress_bar.write(
            f"Topic {bundle.index:04d} finished with {written} translations."
        )

    return written, failures


async def run_translate(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    bundles = load_answer_bundles(
        args.input_dir,
        topic_offset=args.topic_offset,
        topic_limit=args.topic_limit,
    )
    if not bundles:
        print("No answer bundles found; nothing to translate.")
        return

    instruction = load_instruction_blocks(args.instruction_path)
    balinese_dict = load_dictionary(args.balinese_dict)
    cirebonese_dict = load_dictionary(args.cirebonese_dict)

    cfg = TranslationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        model=args.model,
        extra_body=args.extra_body_payload,
        context_limit=args.context_limit,
    )

    tokenize = load_chat_encoder(cfg.model, args.tokenizer_model)

    skip_ids = gather_skip_ids(args.output_dir, args.aggregate_translations)
    skip_lock = asyncio.Lock()

    client = build_client(args)
    request_semaphore = asyncio.Semaphore(max(1, args.request_concurrency))
    topic_semaphore = asyncio.Semaphore(max(1, args.topic_concurrency))

    total_written = 0
    total_failures: List[Tuple[str, str]] = []

    progress_bar = tqdm(total=len(bundles), desc="Topics", unit="topic")

    async def topic_worker(bundle: TopicAnswerBundle) -> None:
        nonlocal total_written
        async with topic_semaphore:
            written, failures = await process_topic(
                bundle,
                client,
                cfg,
                request_semaphore,
                args.output_dir,
                skip_ids,
                skip_lock,
                progress_bar,
                instruction,
                balinese_dict,
                cirebonese_dict,
                tokenize,
            )
            total_written += written
            total_failures.extend(failures)
            progress_bar.update(1)

    await asyncio.gather(*(topic_worker(bundle) for bundle in bundles))
    progress_bar.close()

    await client.close()

    print(f"Completed translation with {total_written} new entries.")
    if total_failures:
        print(
            f"Encountered {len(total_failures)} failures. See logs above for details."
        )


async def run_sample(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    topic_entries = list(
        iter_topic_paths(
            args.input_dir,
            topic_offset=args.topic_offset,
            topic_limit=args.topic_limit,
        )
    )
    if not topic_entries:
        print("No topic files available for sampling.")
        return

    rng = (
        random.Random(args.sample_seed)
        if args.sample_seed is not None
        else random.Random()
    )
    rng.shuffle(topic_entries)

    aggregate_ids: Set[str]
    if args.aggregate_translations:
        aggregate_ids = read_custom_ids_from_jsonl(args.aggregate_translations)
    else:
        aggregate_ids = set()

    translated_cache: Dict[int, Set[str]] = {}

    def is_translated(item: AnswerItem) -> bool:
        if item.custom_id in aggregate_ids:
            return True
        cache = translated_cache.get(item.topic_index)
        if cache is None:
            path = topic_output_path(args.output_dir, item.topic_index)
            cache = read_custom_ids_from_jsonl(path)
            translated_cache[item.topic_index] = cache
        return item.custom_id in cache

    cfg = TranslationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        model=args.model,
        extra_body=args.extra_body_payload,
        context_limit=args.context_limit,
    )

    tokenize = load_chat_encoder(cfg.model, args.tokenizer_model)

    pending: List[AnswerItem] = []
    for topic_index, path in topic_entries:
        items = read_topic_items(path, topic_index)
        if not items:
            continue
        rng.shuffle(items)
        for item in items:
            if args.sample_mode == "untranslated" and is_translated(item):
                continue
            if args.sample_mode == "translated" and not is_translated(item):
                continue
            pending.append(item)
            if len(pending) >= args.count:
                break
        if len(pending) >= args.count:
            break

    if not pending:
        if args.sample_mode == "translated":
            print("No translated answers available for sampling.")
        elif args.sample_mode == "untranslated":
            print("No untranslated answers available for sampling.")
        else:
            print("No answers available for sampling.")
        return

    instruction = load_instruction_blocks(args.instruction_path)
    balinese_dict = load_dictionary(args.balinese_dict)
    cirebonese_dict = load_dictionary(args.cirebonese_dict)

    client = build_client(args)
    request_semaphore = asyncio.Semaphore(max(1, args.request_concurrency))

    sample_output: Path = args.sample_output
    records: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    async def worker(item: AnswerItem) -> None:
        balinese_lexicon = build_lexicon_block(item.answer, balinese_dict)
        cirebonese_lexicon = build_lexicon_block(item.answer, cirebonese_dict)
        prompt = render_prompt(
            instruction, item.answer, balinese_lexicon, cirebonese_lexicon
        )
        prompt_tokens = (
            count_prompt_tokens(DEFAULT_SYSTEM_PROMPT, prompt, tokenize) + 100
        )
        available_tokens = cfg.context_limit - prompt_tokens
        if available_tokens < MIN_COMPLETION_TOKENS:
            failures.append(
                (
                    item.custom_id,
                    f"Insufficient token budget: prompt={prompt_tokens}, limit={cfg.context_limit}",
                )
            )
            return
        max_tokens = min(cfg.max_tokens, available_tokens)

        try:
            completion = await create_chat_completion(
                client,
                prompt,
                cfg,
                request_semaphore,
                max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((item.custom_id, str(exc)))
            return

        message = completion.choices[0].message
        raw_content = message.content if message and message.content else ""
        parsed: Optional[Dict[str, str]] = None
        parse_error: Optional[str] = None
        try:
            parsed = parse_translation_json(raw_content)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            failures.append((item.custom_id, f"Stored with parse error: {parse_error}"))

        record = build_translation_record(
            item,
            prompt,
            balinese_lexicon,
            cirebonese_lexicon,
            completion,
            parsed,
            prompt_tokens=prompt_tokens,
            planned_max_tokens=max_tokens,
            context_limit=cfg.context_limit,
            raw_response=raw_content if parse_error else None,
            parse_error=parse_error,
        )
        records.append(record)

    await asyncio.gather(*(worker(item) for item in pending))
    await client.close()

    if records:
        ensure_directory(sample_output.parent)
        for record in records:
            append_jsonl(sample_output, record)
        print(f"Wrote {len(records)} sample translation record(s) to {sample_output}.")
    else:
        print("No sample records generated.")

    if failures:
        print(f"Sample completed with {len(failures)} failures:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")


def parse_args() -> argparse.Namespace:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--input-dir",
        type=Path,
        default=Path("topic_answers"),
        help="Directory containing per-topic answer JSONL files to translate.",
    )
    parent.add_argument(
        "--output-dir",
        type=Path,
        default=Path("topic_translations"),
        help="Directory to store per-topic translation JSONL files.",
    )
    parent.add_argument(
        "--instruction-path",
        type=Path,
        default=Path("instruction.txt"),
        help="File containing the translation instruction template.",
    )
    parent.add_argument(
        "--balinese-dict",
        type=Path,
        default=Path("dict/idn_bali.json"),
        help="Path to the Indonesian-Balinese lexicon JSON.",
    )
    parent.add_argument(
        "--cirebonese-dict",
        type=Path,
        default=Path("dict/idn_cbn.json"),
        help="Path to the Indonesian-Cirebonese lexicon JSON.",
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
        "--tokenizer-model",
        help=(
            "Optional Hugging Face tokenizer identifier or local path used for token counting. "
            "If omitted, tiktoken will attempt to infer an encoding from --model."
        ),
    )
    parent.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum completion tokens for each translation.",
    )
    parent.add_argument(
        "--context-limit",
        type=int,
        default=MODEL_CONTEXT_LIMIT,
        help="Maximum total tokens (prompt + completion) permitted by the model context window.",
    )
    parent.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for translation requests.",
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
        "--topic-offset",
        type=int,
        default=0,
        help="Skip all topic files with an index lower than this value.",
    )
    parent.add_argument(
        "--topic-limit",
        type=int,
        help="Limit the number of topic files processed in this run.",
    )
    parent.add_argument(
        "--aggregate-translations",
        type=Path,
        help="Optional aggregated translation JSONL whose entries should be skipped.",
    )
    parent.add_argument(
        "--extra-body",
        help="Optional JSON string merged into the OpenAI extra_body payload.",
    )

    parser = argparse.ArgumentParser(
        description="Translate answers into Balinese and Cirebonese via vLLM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate_parser = subparsers.add_parser(
        "translate",
        parents=[parent],
        help="Translate all pending answers into Balinese and Cirebonese.",
    )
    translate_parser.set_defaults(func=lambda args: asyncio.run(run_translate(args)))

    sample_parser = subparsers.add_parser(
        "sample",
        parents=[parent],
        help="Run a sample translation without writing outputs.",
    )
    sample_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of answers to fetch during the sample run.",
    )
    sample_parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("sample.jsonl"),
        help="Path to append JSONL sample outputs (default: sample.jsonl).",
    )
    sample_parser.add_argument(
        "--sample-mode",
        choices=("untranslated", "any", "translated"),
        default="untranslated",
        help="Choose whether to sample only untranslated entries, only already translated entries,"
        " or any entry regardless of status.",
    )
    sample_parser.add_argument(
        "--sample-seed",
        type=int,
        help="Optional random seed to make sampling order deterministic.",
    )
    sample_parser.set_defaults(func=lambda args: asyncio.run(run_sample(args)))

    args = parser.parse_args()
    args.extra_body_payload = parse_extra_body(args.extra_body)
    if args.topic_offset < 0:
        parser.error("--topic-offset must be zero or greater")
    if args.topic_limit is not None and args.topic_limit <= 0:
        parser.error("--topic-limit must be a positive integer")
    if args.context_limit <= 0:
        parser.error("--context-limit must be a positive integer")
    if args.tokenizer_model:
        trimmed = args.tokenizer_model.strip()
        args.tokenizer_model = trimmed or None
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
