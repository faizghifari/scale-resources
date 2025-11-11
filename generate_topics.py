"""Seed topic expansion pipeline using OpenRouter or self-hosted vLLM via the OpenAI SDK.

This script reads seed topics from a plain-text file, generates a fixed number
of subtopics for each seed topic, and then expands every subtopic into a set of
questions. All generations are written to a JSON file and the process can be
resumed safely. To support very large workloads (10k+ topics), the pipeline
uses asyncio with configurable concurrency controls.

Usage example:

    python generate_topics.py \
        --input seed_topics_first_1000.txt \
        --output generated_topics.json \
        --topic-concurrency 4 \
        --request-concurrency 16

To use a local vLLM deployment instead of OpenRouter:

    python generate_topics.py \
        --model-source vllm \
        --api-base http://localhost:8000/v1 \
        --api-key token-abc123 \
        --model NousResearch/Meta-Llama-3-8B-Instruct

By default the script targets the OpenRouter ``qwen/qwen3-235b-a22b:free``
model and expects an ``OPENROUTER_API_KEY`` entry inside ``.env`` (or already
available in the environment). Alternate backends such as a self-hosted vLLM
instance can be selected via command-line flags that set the base URL, API key,
and model identifier. When targeting vLLM, you may set ``VLLM_API_KEY`` or pass
``--api-key`` (defaulting to ``token-abc123`` if not provided).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, cast

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tqdm import tqdm

DEFAULT_SUBTOPICS_PER_TOPIC = 30
DEFAULT_QUESTIONS_PER_SUBTOPIC = 30
DEFAULT_TOPIC_CONCURRENCY = 1
DEFAULT_REQUEST_CONCURRENCY = 30
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "qwen/qwen3-235b-a22b:free"


@dataclass
class GenerationConfig:
    """Configuration knobs for prompt generation and concurrency."""

    subtopics_per_topic: int = DEFAULT_SUBTOPICS_PER_TOPIC
    questions_per_subtopic: int = DEFAULT_QUESTIONS_PER_SUBTOPIC
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 4
    request_delay: float = 0.0
    request_concurrency: int = DEFAULT_REQUEST_CONCURRENCY
    topic_concurrency: int = DEFAULT_TOPIC_CONCURRENCY
    model: str = DEFAULT_MODEL_NAME
    extra_body: Dict[str, Any] = field(default_factory=dict)


def load_seed_topics(path: str) -> List[str]:
    """Load seed topics, stripping blanks and de-duplicating while preserving order."""

    seen = set()
    topics: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            topic = raw_line.strip()
            if not topic or topic in seen:
                continue
            seen.add(topic)
            topics.append(topic)
    return topics


def load_existing_output(output_path: str) -> Dict[str, dict]:
    """Load existing JSON output (list of topic records) if available."""

    if not os.path.exists(output_path):
        return {}

    with open(output_path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to parse existing output JSON at '{output_path}': {exc}"
            ) from exc

    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected output JSON to be a list but received {type(data).__name__}."
        )

    records: Dict[str, dict] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        topic = entry.get("topic")
        if isinstance(topic, str):
            records[topic] = entry
    return records


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    unique_list: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        unique_list.append(normalized)
    return unique_list


def extract_json_payload(raw: str) -> dict:
    """Attempt to parse the assistant response as JSON."""

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("` ")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    first_brace = raw.find("{")
    first_bracket = raw.find("[")
    if first_brace == -1 and first_bracket == -1:
        raise json.JSONDecodeError("No JSON object could be decoded", raw, 0)

    if first_brace == -1 or (first_bracket != -1 and first_bracket < first_brace):
        json_text = raw[first_bracket : raw.rfind("]") + 1]
        wrapped = f'{{"items": {json_text}}}'
        return json.loads(wrapped)

    last_brace = raw.rfind("}")
    json_text = raw[first_brace : last_brace + 1]
    return json.loads(json_text)


async def chat_completion(
    client: AsyncOpenAI,
    messages: Sequence[ChatCompletionMessageParam],
    cfg: GenerationConfig,
    max_tokens: int,
    request_semaphore: asyncio.Semaphore,
) -> ChatCompletion:
    """Execute a chat completion call with retry, delay, and concurrency limits."""

    backoff_base = 2.0
    last_error: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            async with request_semaphore:
                response = await client.chat.completions.create(
                    model=cfg.model,
                    messages=list(messages),
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=max_tokens,
                    reasoning_effort="low",
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
            jitter = random.uniform(0, 0.5)
            await asyncio.sleep(sleep_for + jitter)

    raise RuntimeError("Unreachable state in chat_completion") from last_error


async def generate_subtopics(
    client: AsyncOpenAI,
    topic: str,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
) -> List[str]:
    """Generate a list of unique subtopics for the given seed topic."""

    system_prompt = (
        "You are an expert curriculum designer expanding high-level topics into "
        "concise, distinct subtopics. Always answer with strict JSON "
        "and the subtopics must be in Indonesian language."
    )
    user_prompt = (
        f"Seed topic: {topic}\n"
        f"Produce exactly {cfg.subtopics_per_topic} unique subtopics that explore "
        "different angles, eras, geographies, stakeholders, or applications."
        " Each subtopic should be under 120 characters and avoid numbering."
        '\nRespond ONLY with JSON in the shape: {"subtopics": ["..."]}. /no_think'
    )

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for _ in range(cfg.max_retries):
        try:
            response = await chat_completion(
                client,
                messages,
                cfg,
                max_tokens=2048,
                request_semaphore=request_semaphore,
            )
            content = response.choices[0].message.content or ""
            payload = extract_json_payload(content)
            subtopics_raw = payload.get("subtopics") or payload.get("items")
            if not isinstance(subtopics_raw, list):
                raise ValueError("JSON payload missing 'subtopics' list")
            subtopics = unique_preserve_order(str(item) for item in subtopics_raw)
            if len(subtopics) >= cfg.subtopics_per_topic:
                return subtopics[: cfg.subtopics_per_topic]
            last_error = ValueError("Assistant returned too few subtopics")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to generate subtopics for topic: {topic}"
    ) from last_error


async def generate_questions(
    client: AsyncOpenAI,
    topic: str,
    subtopic: str,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
) -> List[str]:
    """Generate a list of questions for a given subtopic."""

    system_prompt = (
        "You are an investigative interviewer crafting thought-provoking "
        "questions in Indonesian and English contexts. Respond strictly with JSON "
        "and the questions must be in Indonesian language."
    )
    user_prompt = (
        f"Seed topic: {topic}\nSubtopic: {subtopic}\n"
        f"Produce exactly {cfg.questions_per_subtopic} unique questions that probe "
        "details, implications, comparisons, or future directions."
        " Each question should be 30-160 characters, avoid numbering, and be "
        "self-contained.\n"
        'Respond ONLY with JSON in the shape: {"questions": ["..."]}. /no_think'
    )

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None

    for _ in range(cfg.max_retries):
        try:
            response = await chat_completion(
                client,
                messages,
                cfg,
                max_tokens=3584,
                request_semaphore=request_semaphore,
            )
            content = response.choices[0].message.content or ""
            payload = extract_json_payload(content)
            questions_raw = payload.get("questions") or payload.get("items")
            if not isinstance(questions_raw, list):
                raise ValueError("JSON payload missing 'questions' list")
            questions = unique_preserve_order(str(item) for item in questions_raw)
            if len(questions) >= cfg.questions_per_subtopic:
                return questions[: cfg.questions_per_subtopic]
            last_error = ValueError("Assistant returned too few questions")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to generate questions for subtopic: {subtopic}"
    ) from last_error


def build_topic_record(
    topic: str,
    subtopics: Sequence[str],
    questions_per_subtopic: Sequence[Sequence[str]],
) -> dict:
    return {
        "topic": topic,
        "subtopics": [
            {
                "title": subtopic,
                "questions": list(questions),
            }
            for subtopic, questions in zip(
                subtopics, questions_per_subtopic, strict=True
            )
        ],
    }


def ordered_records(
    topic_order: Mapping[str, int],
    records: Mapping[str, dict],
) -> List[dict]:
    return [
        records[topic]
        for topic in sorted(
            records.keys(),
            key=lambda item: (topic_order.get(item, float("inf")), item),
        )
    ]


def write_output(
    output_path: str,
    topic_order: Mapping[str, int],
    records: Mapping[str, dict],
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            ordered_records(topic_order, records), handle, ensure_ascii=False, indent=2
        )
        handle.write("\n")


async def write_output_async(
    output_path: str,
    topic_order: Mapping[str, int],
    records: Mapping[str, dict],
) -> None:
    await asyncio.to_thread(write_output, output_path, topic_order, dict(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand seed topics into subtopics and questions using OpenRouter with "
            "configurable concurrency."
        ),
    )
    parser.add_argument(
        "--input",
        default="seed_topics_first_1000.txt",
        help="Path to plaintext file containing one seed topic per line.",
    )
    parser.add_argument(
        "--output",
        default="generated_topics.json",
        help="Destination JSON file for storing generations.",
    )
    parser.add_argument(
        "--subtopics-per-topic",
        type=int,
        default=DEFAULT_SUBTOPICS_PER_TOPIC,
        help="Number of subtopics to generate for each seed topic.",
    )
    parser.add_argument(
        "--questions-per-subtopic",
        type=int,
        default=DEFAULT_QUESTIONS_PER_SUBTOPIC,
        help="Number of questions to generate for each subtopic.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Limit the number of seed topics to process (for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of resuming from it.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each API call to stay within rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum attempts for each generation step before failing.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--topic-concurrency",
        type=int,
        default=DEFAULT_TOPIC_CONCURRENCY,
        help="Maximum number of seed topics to process concurrently.",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=DEFAULT_REQUEST_CONCURRENCY,
        help="Maximum number of simultaneous API requests.",
    )
    parser.add_argument(
        "--model-source",
        choices=["openrouter", "vllm"],
        default="openrouter",
        help="Model backend to target (OpenRouter or self-hosted vLLM).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model identifier to request from the selected backend.",
    )
    parser.add_argument(
        "--api-base",
        help="Override the API base URL for the selected backend.",
    )
    parser.add_argument(
        "--api-key",
        help="Override the API key/token for the selected backend.",
    )
    parser.add_argument(
        "--extra-body",
        help="Optional JSON string merged into the OpenAI extra_body payload.",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> AsyncOpenAI:
    load_dotenv()
    source = args.model_source

    if source == "openrouter":
        base_url = args.api_base or OPENROUTER_BASE_URL
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Provide --api-key or set it in the environment."
            )
    else:
        base_url = args.api_base or DEFAULT_VLLM_BASE_URL
        api_key = args.api_key or os.getenv("VLLM_API_KEY") or "token-abc123"

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def process_topic(
    topic: str,
    client: AsyncOpenAI,
    cfg: GenerationConfig,
    request_semaphore: asyncio.Semaphore,
    records: MutableMapping[str, dict],
    topic_order: Mapping[str, int],
    output_path: str,
    records_lock: asyncio.Lock,
    progress_bar: tqdm,
    progress_lock: asyncio.Lock,
    failures: List[tuple[str, str]],
) -> None:
    try:
        subtopics = await generate_subtopics(client, topic, cfg, request_semaphore)

        question_tasks = [
            asyncio.create_task(
                generate_questions(client, topic, subtopic, cfg, request_semaphore)
            )
            for subtopic in subtopics
        ]
        question_results = await asyncio.gather(*question_tasks, return_exceptions=True)

        successful_subtopics: List[str] = []
        questions_bundle: List[List[str]] = []
        subtopic_failures: List[tuple[str, str]] = []

        for subtopic, result in zip(subtopics, question_results):
            if isinstance(result, BaseException):
                subtopic_failures.append((subtopic, str(result)))
                continue
            questions_bundle.append(cast(List[str], result))
            successful_subtopics.append(subtopic)

        if not successful_subtopics:
            error_reason = subtopic_failures[-1][1] if subtopic_failures else "unknown"
            raise RuntimeError(
                f"All subtopics failed for '{topic}'. Last error: {error_reason}"
            )

        if subtopic_failures:
            summary = "; ".join(f"{name}: {err}" for name, err in subtopic_failures[:3])
            async with progress_lock:
                progress_bar.write(
                    f"Topic '{topic}' skipped {len(subtopic_failures)} subtopics due to errors: {summary}"
                )

        record = build_topic_record(topic, successful_subtopics, questions_bundle)

        async with records_lock:
            records[topic] = record
            await write_output_async(output_path, topic_order, records)

        async with progress_lock:
            progress_bar.set_postfix_str(topic[:40])
            progress_bar.update(1)

    except Exception as exc:  # noqa: BLE001
        async with progress_lock:
            progress_bar.write(f"Failed to process '{topic}': {exc}")
            progress_bar.update(1)
        async with records_lock:
            failures.append((topic, str(exc)))


async def run_pipeline(args: argparse.Namespace, cfg: GenerationConfig) -> None:
    topics = load_seed_topics(args.input)
    if args.max_topics is not None:
        topics = topics[: args.max_topics]

    topic_order = {topic: idx for idx, topic in enumerate(topics)}

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
        records: MutableMapping[str, dict] = {}
    else:
        records = load_existing_output(args.output)
        for topic in records:
            if topic not in topic_order:
                topic_order[topic] = len(topic_order)

    pending_topics = [topic for topic in topics if topic not in records]
    if not pending_topics:
        tqdm.write("No pending topics detected — everything is already processed.")
        return

    if records:
        tqdm.write(f"Resuming from existing JSON with {len(records)} completed topics.")

    client = build_client(args)
    request_semaphore = asyncio.Semaphore(max(1, cfg.request_concurrency))
    topic_semaphore = asyncio.Semaphore(max(1, cfg.topic_concurrency))
    records_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    failures: List[tuple[str, str]] = []

    progress_bar = tqdm(total=len(pending_topics), desc="Seed topics", unit="topic")

    async def topic_worker(topic: str) -> None:
        async with topic_semaphore:
            await process_topic(
                topic,
                client,
                cfg,
                request_semaphore,
                records,
                topic_order,
                args.output,
                records_lock,
                progress_bar,
                progress_lock,
                failures,
            )

    await asyncio.gather(*(topic_worker(topic) for topic in pending_topics))

    progress_bar.close()
    await client.close()

    if failures:
        tqdm.write(
            f"Completed with {len(failures)} failures. See log above for details."
        )


def main() -> None:
    args = parse_args()
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid JSON for --extra-body: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object")
    elif args.model_source == "openrouter":
        extra_body = {"reasoning": {"enabled": False}}
    else:
        extra_body = {}

    cfg = GenerationConfig(
        subtopics_per_topic=args.subtopics_per_topic,
        questions_per_subtopic=args.questions_per_subtopic,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
        request_concurrency=args.request_concurrency,
        topic_concurrency=args.topic_concurrency,
        model=args.model,
        extra_body=extra_body,
    )

    try:
        asyncio.run(run_pipeline(args, cfg))
    except KeyboardInterrupt:
        tqdm.write("Interrupted by user")


if __name__ == "__main__":
    main()
