#!/usr/bin/env python
"""Generate synthetic answers for topic questions using the OpenAI Batch API.

The script reads questions from a ``generated_topics.json`` file (produced by
``generate_topics.py``), prepares request batches for the ``gpt-5-nano`` model,
and offers utilities to create, submit, monitor, and retrieve batch jobs.
When a batch completes, responses can be merged into an aggregate JSONL file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from dotenv import load_dotenv
from openai import OpenAI

BATCH_MODEL = "gpt-5-nano"
BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant who always responds in fluent, natural Indonesian. "
    "Deliver only the final answer; never describe internal reasoning or thought processes. "
    "Reduce the use of bullet points, numbered lists, headings, or any list-like formatting; only use it when really necessary. "
    "Vary sentence length and tone to keep the prose engaging, weaving in relevant examples, comparisons, or brief illustrative anecdotes when helpful. "
    "Do not repeat or rephrase the question; begin directly with the answer in Indonesian language."
)

CompletionWindow = Literal["24h"]
T = TypeVar("T")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunk_iterable(items: List[T], size: int) -> Iterable[List[T]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return cast(Dict[str, Any], json.load(handle))


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    return read_json_file(meta_path)


def update_metadata(meta_path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    current = load_metadata(meta_path)
    current.update(updates)
    write_json_file(meta_path, current)
    return current


def require_api_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Please set it in .env or the environment."
        )


def _download_file(client: OpenAI, file_id: str, destination: Path) -> bytes:
    response = client.files.content(file_id)
    raw_bytes = response.read()
    destination.write_bytes(raw_bytes)
    return raw_bytes


def _resolve_batch_inputs(
    batch_id: Optional[str], meta_path: Optional[Path]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    meta: Optional[Dict[str, Any]] = None
    if meta_path:
        meta = load_metadata(meta_path)
        batch_id = batch_id or cast(Optional[str], meta.get("batch_id"))
    if not batch_id:
        raise RuntimeError("Batch ID is required via --batch-id or metadata file.")
    return batch_id, meta


@dataclass
class QuestionItem:
    custom_id: str
    topic: str
    subtopic: str
    question: str


def load_questions(input_path: Path) -> List[QuestionItem]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    items: List[QuestionItem] = []
    for topic_idx, topic_entry in enumerate(data):
        topic = topic_entry.get("topic", "")
        for sub_idx, sub_entry in enumerate(topic_entry.get("subtopics", [])):
            subtopic = sub_entry.get("title", "")
            for question_idx, question in enumerate(sub_entry.get("questions", [])):
                custom_id = f"topic{topic_idx:04d}-sub{sub_idx:03d}-q{question_idx:03d}"
                items.append(
                    QuestionItem(
                        custom_id=custom_id,
                        topic=topic,
                        subtopic=subtopic,
                        question=str(question),
                    )
                )
    return items


def write_batch_requests(items: Iterable[QuestionItem], jsonl_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in items:
            request_body = {
                "custom_id": item.custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": {
                    "model": BATCH_MODEL,
                    "max_completion_tokens": 1024,
                    "reasoning_effort": "minimal",
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": item.question},
                    ],
                },
            }
            handle.write(json.dumps(request_body, ensure_ascii=False) + "\n")


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return cast(Dict[str, Any], usage.model_dump())
    if isinstance(usage, dict):
        return usage
    try:
        return dict(usage)  # type: ignore[arg-type]
    except TypeError:
        return {}


def _extract_usage_fields(usage: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    if not usage:
        return None, None, None
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("total_prompt_tokens")
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("total_completion_tokens")
    )
    reasoning_tokens = usage.get("reasoning_tokens")
    if reasoning_tokens is None:
        details = usage.get("output_tokens_details") or usage.get(
            "completion_tokens_details"
        )
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens") or (
                details.get("reasoning") or {}
            ).get("tokens")
    return prompt_tokens, completion_tokens, reasoning_tokens


def load_existing_answers(output_path: Path) -> Tuple[Dict[str, Dict], List[str]]:
    if not output_path.exists():
        return {}, []

    records: List[Dict] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "custom_id" not in record:
                continue
            records.append(record)

    record_map: Dict[str, Dict] = {}
    order: List[str] = []
    for record in records:
        cid = record["custom_id"]
        if cid in record_map:
            continue
        record_map[cid] = record
        order.append(cid)
    return record_map, order


def add_answer_record(
    record_map: Dict[str, Dict],
    order: List[str],
    record: Dict,
) -> None:
    cid = record.get("custom_id")
    if not cid:
        return
    if cid not in record_map:
        order.append(cid)
    record_map[cid] = record


def create_batch(
    client: OpenAI, request_file: Path, completion_window: CompletionWindow
) -> Tuple[str, str, Dict[str, Any]]:
    with request_file.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=completion_window,
        metadata={"source": "generate_synthetic_answers"},
    )
    return batch.id, file_obj.id, batch.model_dump()


def merge_answers(
    responses: Iterable[Dict],
    question_lookup: Dict[str, QuestionItem],
) -> List[Dict]:
    merged: List[Dict] = []
    for entry in responses:
        custom_id = entry.get("custom_id")
        payload = entry.get("response", {})
        body = payload.get("body", {})
        choices = body.get("choices", [])
        if not custom_id or not choices:
            continue
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        question = question_lookup.get(custom_id)
        if question is None:
            continue
        usage_raw = _usage_to_dict(body.get("usage"))
        prompt_tokens, completion_tokens, reasoning_tokens = _extract_usage_fields(
            usage_raw
        )
        merged.append(
            {
                "custom_id": custom_id,
                "topic": question.topic,
                "subtopic": question.subtopic,
                "question": question.question,
                "answer": content,
                "model": body.get("model", BATCH_MODEL),
                **(
                    {"prompt_tokens": prompt_tokens}
                    if prompt_tokens is not None
                    else {}
                ),
                **(
                    {"completion_tokens": completion_tokens}
                    if completion_tokens is not None
                    else {}
                ),
                **(
                    {"reasoning_tokens": reasoning_tokens}
                    if reasoning_tokens is not None
                    else {}
                ),
                **({"usage": usage_raw} if usage_raw else {}),
            }
        )
    return merged


def write_outputs(answers: Iterable[Dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")


def run_sample_answers(
    client: OpenAI,
    questions: Iterable[QuestionItem],
) -> List[Dict]:
    samples: List[Dict] = []
    chat_completions = cast(Any, client.chat.completions)
    for item in questions:
        response = chat_completions.create(
            model=BATCH_MODEL,
            max_completion_tokens=1024,
            reasoning_effort="minimal",
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": item.question},
            ],
        )
        response_payload = response.model_dump()
        content = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage_raw = _usage_to_dict(response_payload.get("usage"))
        prompt_tokens, completion_tokens, reasoning_tokens = _extract_usage_fields(
            usage_raw
        )
        record = {
            "custom_id": item.custom_id,
            "topic": item.topic,
            "subtopic": item.subtopic,
            "question": item.question,
            "answer": str(content).strip(),
            "model": response_payload.get("model") or BATCH_MODEL,
            **({"prompt_tokens": prompt_tokens} if prompt_tokens is not None else {}),
            **(
                {"completion_tokens": completion_tokens}
                if completion_tokens is not None
                else {}
            ),
            **(
                {"reasoning_tokens": reasoning_tokens}
                if reasoning_tokens is not None
                else {}
            ),
            **({"usage": usage_raw} if usage_raw else {}),
        }
        samples.append(record)
        preview = record["answer"].replace("\n", " ")[:120]
        print(f"Sample answer for {item.custom_id}: {preview}...")
    return samples


def handle_create(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    answers_path = Path(args.answers)
    output_dir = Path(args.output_dir)

    questions = load_questions(input_path)
    answer_map, _ = load_existing_answers(answers_path)
    unanswered = [q for q in questions if q.custom_id not in answer_map]

    if not unanswered:
        print("All questions already answered; no batch files created.")
        return

    ensure_directory(output_dir)
    chunk_size = max(1, args.chunk_size)
    created = 0
    skipped = 0

    for index, chunk in enumerate(chunk_iterable(unanswered, chunk_size)):
        first_id = chunk[0].custom_id
        last_id = chunk[-1].custom_id
        base_name = f"batch_{index:04d}_{first_id}_to_{last_id}"
        jsonl_path = output_dir / f"{base_name}.jsonl"
        meta_path = jsonl_path.with_suffix(".meta.json")

        if jsonl_path.exists() and not args.overwrite:
            print(
                f"Skipping existing batch file {jsonl_path} (use --overwrite to regenerate)."
            )
            skipped += 1
            continue

        write_batch_requests(chunk, jsonl_path)
        meta_payload = {
            "status": "created",
            "created_at": now_utc_iso(),
            "jsonl_path": str(jsonl_path.resolve()),
            "question_ids": [item.custom_id for item in chunk],
            "input_path": str(input_path.resolve()),
            "answers_path": str(answers_path.resolve()),
            "chunk_size": chunk_size,
            "batch_id": None,
            "input_file_id": None,
            "output_file_id": None,
            "error_file_id": None,
        }
        write_json_file(meta_path, meta_payload)
        print(f"Wrote {jsonl_path} with {len(chunk)} requests. Metadata: {meta_path}")
        created += 1

    print(
        f"Prepared {created} batch files in {output_dir} (skipped {skipped} existing files)."
    )


def handle_sample(args: argparse.Namespace) -> None:
    require_api_key()
    input_path = Path(args.input)
    answers_path = Path(args.answers)

    questions = load_questions(input_path)
    answer_map, answer_order = load_existing_answers(answers_path)
    unanswered = [q for q in questions if q.custom_id not in answer_map]

    if not unanswered:
        print("All questions already answered; nothing to sample.")
        return

    sample_count = max(0, args.count)
    if sample_count == 0:
        print("Sample count is zero; nothing to do.")
        return

    client = OpenAI()
    sample_questions = unanswered[:sample_count]
    sample_answers = run_sample_answers(client, sample_questions)
    for record in sample_answers:
        add_answer_record(answer_map, answer_order, record)

    write_outputs([answer_map[cid] for cid in answer_order], answers_path)
    print(
        f"Wrote {len(answer_order)} answers to {answers_path} "
        f"({len(sample_answers)} via sample)."
    )


def handle_submit(args: argparse.Namespace) -> None:
    require_api_key()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise RuntimeError(f"Batch request file not found: {jsonl_path}")

    meta_path = Path(args.meta) if args.meta else jsonl_path.with_suffix(".meta.json")
    meta = load_metadata(meta_path)
    if not meta:
        meta = {"jsonl_path": str(jsonl_path.resolve())}

    client = OpenAI()
    completion_window = cast(CompletionWindow, args.completion_window)
    batch_id, input_file_id, batch_info = create_batch(
        client, jsonl_path, completion_window
    )

    updated_meta = update_metadata(
        meta_path,
        {
            "batch_id": batch_id,
            "input_file_id": input_file_id,
            "status": batch_info.get("status"),
            "submitted_at": now_utc_iso(),
            "completion_window": completion_window,
            "last_batch_response": batch_info,
            "jsonl_path": str(jsonl_path.resolve()),
        },
    )

    print(
        f"Submitted batch {batch_id} with status {updated_meta.get('status')}. "
        f"Metadata saved to {meta_path}."
    )


def handle_check(args: argparse.Namespace) -> None:
    require_api_key()
    meta_path = Path(args.meta) if args.meta else None
    batch_id, _ = _resolve_batch_inputs(args.batch_id, meta_path)

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    batch_info = batch.model_dump()
    status = batch_info.get("status")
    print(f"Batch {batch_id} status: {status}")

    if meta_path:
        update_metadata(
            meta_path,
            {
                "status": status,
                "output_file_id": batch_info.get("output_file_id"),
                "error_file_id": batch_info.get("error_file_id"),
                "last_batch_response": batch_info,
                "last_checked_at": now_utc_iso(),
            },
        )
        print(f"Metadata updated at {meta_path}.")


def handle_retrieve(args: argparse.Namespace) -> None:
    require_api_key()
    meta_path = Path(args.meta) if args.meta else None
    batch_id, meta = _resolve_batch_inputs(args.batch_id, meta_path)

    answers_path = Path(args.answers)
    input_path = Path(args.input)
    if meta:
        answers_path = Path(meta.get("answers_path", answers_path))
        input_path = Path(meta.get("input_path", input_path))

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    batch_info = batch.model_dump()
    status = batch_info.get("status")
    output_file_id = batch_info.get("output_file_id")
    error_file_id = batch_info.get("error_file_id")

    if meta_path:
        update_metadata(
            meta_path,
            {
                "status": status,
                "output_file_id": output_file_id,
                "error_file_id": error_file_id,
                "last_batch_response": batch_info,
                "last_checked_at": now_utc_iso(),
            },
        )

    print(f"Batch {batch_id} status: {status}")
    if not output_file_id:
        raise RuntimeError("Batch does not yet have an output file; try again later.")

    jsonl_path = Path(meta["jsonl_path"]) if meta and meta.get("jsonl_path") else None
    base_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (jsonl_path.parent if jsonl_path else Path.cwd())
    )
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"
    errors_dir = Path(args.errors_dir) if args.errors_dir else base_dir / "errors"

    ensure_directory(results_dir)

    output_path = results_dir / f"{batch_id}_output.jsonl"
    raw_bytes = _download_file(client, output_file_id, output_path)
    lines = [line for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    responses = [json.loads(line) for line in lines]

    error_path: Optional[Path] = None
    if error_file_id:
        ensure_directory(errors_dir)
        error_path = errors_dir / f"{batch_id}_errors.jsonl"
        _download_file(client, error_file_id, error_path)

    questions = load_questions(input_path)
    question_lookup = {item.custom_id: item for item in questions}
    batch_answers = merge_answers(responses, question_lookup)

    answer_map, answer_order = load_existing_answers(answers_path)
    for record in batch_answers:
        add_answer_record(answer_map, answer_order, record)
    write_outputs([answer_map[cid] for cid in answer_order], answers_path)

    if meta_path:
        update_metadata(
            meta_path,
            {
                "retrieved_at": now_utc_iso(),
                "output_file_path": str(output_path.resolve()),
                "error_file_path": str(error_path.resolve()) if error_path else None,
                "status": status,
            },
        )

    print(
        f"Merged {len(batch_answers)} answers into {answers_path}. "
        f"Results stored at {output_path}."
    )
    if error_path:
        print(f"Error records stored at {error_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage synthetic answer generation batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create", help="Create batch request JSONL files for unanswered questions."
    )
    create_parser.add_argument(
        "--input",
        default="generated_topics.json",
        help="Path to the JSON file containing topics, subtopics, and questions.",
    )
    create_parser.add_argument(
        "--answers",
        default="synthetic_answers.jsonl",
        help="Existing answers JSONL (used to skip already answered questions).",
    )
    create_parser.add_argument(
        "--output-dir",
        default="batches",
        help="Directory to store generated batch JSONL files and metadata.",
    )
    create_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Maximum number of questions per batch JSONL file.",
    )
    create_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch JSONL and metadata files.",
    )
    create_parser.set_defaults(func=handle_create)

    sample_parser = subparsers.add_parser(
        "sample", help="Answer a small number of questions synchronously for review."
    )
    sample_parser.add_argument(
        "--input",
        default="generated_topics.json",
        help="Path to the JSON file containing topics, subtopics, and questions.",
    )
    sample_parser.add_argument(
        "--answers",
        default="synthetic_answers.jsonl",
        help="Destination JSONL file for aggregated answers.",
    )
    sample_parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of unanswered questions to answer synchronously.",
    )
    sample_parser.set_defaults(func=handle_sample)

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a prepared batch JSONL file to the OpenAI Batch API."
    )
    submit_parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to the batch request JSONL file to submit.",
    )
    submit_parser.add_argument(
        "--meta",
        help="Path to the metadata JSON file (defaults to <jsonl>.meta.json).",
    )
    submit_parser.add_argument(
        "--completion-window",
        default="24h",
        choices=["24h"],
        help="Batch completion window supported by this script.",
    )
    submit_parser.set_defaults(func=handle_submit)

    check_parser = subparsers.add_parser(
        "check", help="Check the status of a submitted batch."
    )
    check_parser.add_argument(
        "--batch-id",
        help="Batch ID returned when the batch was submitted.",
    )
    check_parser.add_argument(
        "--meta",
        help="Path to the metadata JSON file associated with the batch.",
    )
    check_parser.set_defaults(func=handle_check)

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Download completed batch results and merge into the answers file.",
    )
    retrieve_parser.add_argument(
        "--batch-id",
        help="Batch ID to retrieve (optional when --meta is provided).",
    )
    retrieve_parser.add_argument(
        "--meta",
        help="Path to the metadata JSON file associated with the batch.",
    )
    retrieve_parser.add_argument(
        "--input",
        default="generated_topics.json",
        help="Path to the JSON file containing topics, subtopics, and questions.",
    )
    retrieve_parser.add_argument(
        "--answers",
        default="synthetic_answers.jsonl",
        help="Destination JSONL file for aggregated answers.",
    )
    retrieve_parser.add_argument(
        "--output-dir",
        help="Base directory for downloaded files. Defaults to the batch JSONL directory.",
    )
    retrieve_parser.add_argument(
        "--results-dir",
        help="Directory to store downloaded result files. Defaults to <base>/results.",
    )
    retrieve_parser.add_argument(
        "--errors-dir",
        help="Directory to store downloaded error files. Defaults to <base>/errors.",
    )
    retrieve_parser.set_defaults(func=handle_retrieve)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
