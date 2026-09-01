#!/usr/bin/env python
"""Normalize heterogeneous Indonesian datasets into a shared schema.

This script loads a curated list of public Indonesian datasets (mostly from the
Hugging Face Hub plus two local SQuAD-style JSON files) and saves each source as
an on-disk Hugging Face Dataset (``save_to_disk``) using a unified,
instruction-following friendly schema. The unified records keep provenance and
task-specific metadata while standardizing the fields required for later
translation and SFT.

Key points:
- Single-row canonical record with `translations` map to be filled later.
- MCQ outputs are stored as "<LETTER>. <option text>" with the letter in
  metadata.answer_key.
- Reasoning traces are retained in metadata.reasoning_raw with tags rewritten to
  <think>…</think> when the source used <Thought>…</Thought>.
- Local SQuAD-style files under dataset/midtraining are supported.

Typical usage:
    python scripts/prepare_unified_dataset.py \
        --output-dir dataset/midtraining_unified

Optional flags:
    --sources oasst1 indonli  # limit to a subset for quick runs
    --max-per-source 1000     # sampling during dry runs
    --streaming               # use datasets streaming when supported
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import tempfile
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # Lazy import so the file can be inspected without datasets installed.
    from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
except Exception:  # pragma: no cover - optional dependency guard
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    IterableDataset = None  # type: ignore
    load_dataset = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency guard
    hf_hub_download = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency guard
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

DEFAULT_SOURCES: Tuple[str, ...] = (
    "oasst1",
    "Ichsan2895/OASST_Top1_Indonesian",
    "izzulgod/indonesian-conversation",
    "FreedomIntelligence/evol-instruct-indonesian",
    "CohereLabs/Global-MMLU",
    "indolem/IndoMMLU",
    "nayeon212/BLEnD",
    "SEACrowd/indoqa",
    "SEACrowd/facqa",
    "SEACrowd/squad_id",
    "izzulgod/indonesian-reasoning",
    "hafidhsoekma/math-olympiad-indonesian-benchmark",
    "afaji/indonli",
    "Deddy/Indonesia-dataset-2023",
    "local_squad_files",
)

SQUAD_LOCAL_FILES = (
    Path("dataset/midtraining/train-squad-v2.0-translated_fixed_enhanced.json"),
    Path(
        "dataset/midtraining/tydiqa-goldp-v1.1-train-indonesian_prepared_enhanced.json"
    ),
)

NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}

THOUGHT_RE = re.compile(r"<Thought>(.*?)</Thought>", re.IGNORECASE | re.DOTALL)
OUTPUT_RE = re.compile(r"<Output>(.*?)</Output>", re.IGNORECASE | re.DOTALL)


def normalize_role(role: str) -> str:
    rl = (role or "").lower()
    if rl in {"assistant", "gpt", "model"}:
        return "assistant"
    if rl in {"human", "user"}:
        return "user"
    if rl == "system":
        return "system"
    return "assistant"


def make_record(
    *,
    uid: str,
    source: str,
    task_type: str,
    role_messages: List[Dict[str, str]],
    output_text: str,
    instruction: Optional[str] = None,
    input_payload: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    return {
        "id": uid,
        "source": source,
        "lang": "id",
        "task_type": task_type,
        "role_messages": role_messages,
        "instruction": instruction or "",
        "input": input_payload,
        "output": output_text,
        "translations": {},
        "metadata": metadata or {},
    }


def build_dataset(
    factory: Callable[[], Iterable[Dict]], log_every: int | None = None, label: str = ""
) -> Dataset:
    if Dataset is None:
        raise RuntimeError("The 'datasets' package is required.")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.jsonl"
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for rec in factory():
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if log_every and count % log_every == 0:
                    logging.info("%s: %d records", label, count)

        if count == 0:
            logging.warning("%s produced no records; writing empty dataset", label)
            return Dataset.from_list([])

        ds = Dataset.from_json(str(path))
        logging.info("%s total rows: %d", label, count)
        return ds


def pick_majority_annotation(annotations: Sequence[Dict]) -> Optional[str]:
    best_answer = None
    best_count = -1
    for ann in annotations:
        answers = ann.get("answers") or []
        if not answers:
            continue
        count = ann.get("count", 1)
        if count > best_count:
            best_count = count
            best_answer = answers[0]
    return best_answer


def parse_reasoning_blocks(text: str) -> Tuple[Optional[str], str]:
    reasoning_match = THOUGHT_RE.search(text)
    output_match = OUTPUT_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    if reasoning:
        reasoning = f"<think>{reasoning}</think>"
    final_answer = output_match.group(1).strip() if output_match else text.strip()
    return reasoning, final_answer


def extract_facqa_answer(tokens: Sequence[str], labels: Sequence[str]) -> str:
    start = None
    end = None
    for idx, tag in enumerate(labels):
        if tag == "B":
            start = idx
            end = idx
            j = idx + 1
            while j < len(labels) and labels[j] == "I":
                end = j
                j += 1
            break
    if start is None or end is None:
        return ""
    return " ".join(tokens[start : end + 1]).replace("  ", " ").strip()


def format_mcq_output(letter: str, options: Dict[str, str]) -> str:
    choice = options.get(letter.upper())
    if choice:
        return f"{letter.upper()}. {choice}"
    return letter.upper()


def iter_local_squad(file_path: Path, source: str) -> Iterator[Dict]:
    if not file_path.exists():
        logging.warning("Local SQuAD file missing: %s", file_path)
        return iter(())
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "data" in raw:
            data = raw.get("data") or []
        elif "paragraphs" in raw:
            # File already represents a single article
            data = [raw]
        else:
            data = []
    elif isinstance(raw, list):
        data = raw
    else:
        data = []
    for article_idx, article in enumerate(data):
        if not isinstance(article, dict):
            continue
        title = article.get("title")
        paragraphs = article.get("paragraphs", [])

        # Case 1: title and paragraphs are dicts keyed by stringified indices; align them.
        if isinstance(title, dict) and isinstance(paragraphs, dict):
            keys = sorted(
                paragraphs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
            )
            for k in keys:
                para_group = paragraphs.get(k, [])
                title_str = title.get(k, "")
                para_list: List[Dict] = []
                if isinstance(para_group, list):
                    para_list = [p for p in para_group if isinstance(p, dict)]
                elif isinstance(para_group, dict):
                    para_list = list(para_group.values())
                for para_idx, para in enumerate(para_list):
                    context = para.get("context", "")
                    for qa in para.get("qas", []):
                        qid = (
                            qa.get("id")
                            or f"{source}:{article_idx}:{k}:{para_idx}:{len(context)}"
                        )
                        question = qa.get("question", "").strip()
                        answers = (
                            qa.get("answers") or qa.get("indonesian_answers") or []
                        )
                        is_impossible = qa.get("is_impossible", False)
                        if answers:
                            ans_text = answers[0].get("text", "").strip()
                            span_start = answers[0].get("answer_start") or answers[
                                0
                            ].get("start_index")
                        else:
                            ans_text = ""
                            span_start = None
                        yield make_record(
                            uid=f"{source}:{qid}",
                            source=source,
                            task_type="qa_span",
                            role_messages=[{"role": "user", "content": question}],
                            output_text=(
                                ans_text if not is_impossible else "unanswerable"
                            ),
                            input_payload={"context": context},
                            metadata={
                                "title": title_str,
                                "span_start": span_start,
                                "is_impossible": is_impossible,
                                "file_name": file_path.name,
                            },
                        )
            continue

        # Case 2: paragraphs as dict/list with a scalar title
        if isinstance(paragraphs, dict):
            try:
                # Preserve numeric order when paragraphs are keyed by indices
                paragraphs = [
                    paragraphs[k]
                    for k in sorted(
                        paragraphs.keys(),
                        key=lambda x: int(x) if str(x).isdigit() else str(x),
                    )
                ]
            except Exception:  # noqa: BLE001
                paragraphs = list(paragraphs.values())
        elif not isinstance(paragraphs, list):
            paragraphs = []
        normalized_paragraphs: List[Dict] = []
        for para in paragraphs:
            if isinstance(para, dict):
                normalized_paragraphs.append(para)
            elif isinstance(para, list):
                normalized_paragraphs.extend([p for p in para if isinstance(p, dict)])
        for para_idx, para in enumerate(normalized_paragraphs):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = (
                    qa.get("id") or f"{source}:{article_idx}:{para_idx}:{len(context)}"
                )
                question = qa.get("question", "").strip()
                answers = qa.get("answers") or qa.get("indonesian_answers") or []
                is_impossible = qa.get("is_impossible", False)
                if answers:
                    ans_text = answers[0].get("text", "").strip()
                    span_start = answers[0].get("answer_start") or answers[0].get(
                        "start_index"
                    )
                else:
                    ans_text = ""
                    span_start = None
                record = make_record(
                    uid=f"{source}:{qid}",
                    source=source,
                    task_type="qa_span",
                    role_messages=[{"role": "user", "content": question}],
                    output_text=ans_text if not is_impossible else "unanswerable",
                    input_payload={"context": context},
                    metadata={
                        "title": title if isinstance(title, str) else "",
                        "span_start": span_start,
                        "is_impossible": is_impossible,
                        "file_name": file_path.name,
                    },
                )
                yield record


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------


def load_oasst1(args: argparse.Namespace) -> Iterator[Dict]:
    try:
        ds_dict = load_dataset("OpenAssistant/oasst1", streaming=args.streaming)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping oasst1: %s", exc)
        return iter(())
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            if row.get("lang") and row.get("lang") != "id":
                continue
            rid = (
                row.get("message_tree_id")
                or row.get("id")
                or f"oasst1:{split_name}:{idx}"
            )
            messages = row.get("messages") or []
            role_messages = [
                {
                    "role": normalize_role(m.get("role", "assistant")),
                    "content": m.get("text", "").strip(),
                }
                for m in messages
                if m.get("text")
            ]
            if not role_messages:
                continue
            output_text = next(
                (
                    m["content"]
                    for m in reversed(role_messages)
                    if m["role"] == "assistant"
                ),
                "",
            )
            yield make_record(
                uid=f"oasst1:{rid}",
                source="oasst1",
                task_type="chat",
                role_messages=role_messages,
                output_text=output_text,
                metadata={"split": split_name},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_ichsan_top1(args: argparse.Namespace) -> Iterator[Dict]:
    ds_dict = load_dataset("Ichsan2895/OASST_Top1_Indonesian", streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            uid = row.get("Unnamed:") or row.get("id") or f"ichsan:{split_name}:{idx}"
            instr = row.get("instruction_id") or row.get("instruction") or ""
            out = row.get("output_id") or row.get("output") or ""
            role_messages = [
                {"role": "user", "content": instr.strip()},
                {"role": "assistant", "content": out.strip()},
            ]
            yield make_record(
                uid=f"ichsan:{uid}",
                source="Ichsan2895/OASST_Top1_Indonesian",
                task_type="chat",
                role_messages=role_messages,
                output_text=out.strip(),
                metadata={"split": split_name},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_izzulgod_conversation(args: argparse.Namespace) -> Iterator[Dict]:
    try:
        ds_dict = load_dataset("ChavyvAkvar/indonesian-conversation-Converted", streaming=args.streaming)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping indonesian-conversation: %s", exc)
        return iter(())
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            uid = row.get("id") or f"izzulgod_conv:{split_name}:{idx}"
            messages = row.get("messages") or []
            role_messages = [
                {
                    "role": normalize_role(m.get("role", "assistant")),
                    "content": (m.get("content") or "").strip(),
                }
                for m in messages
                if m.get("content")
            ]
            if not role_messages:
                continue
            output_text = next(
                (
                    m["content"]
                    for m in reversed(role_messages)
                    if m["role"] == "assistant"
                ),
                "",
            )
            yield make_record(
                uid=f"izzulgod_conv:{uid}",
                source="izzulgod/indonesian-conversation",
                task_type="chat",
                role_messages=role_messages,
                output_text=output_text,
                metadata={"split": split_name},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_evol_instruct(args: argparse.Namespace) -> Iterator[Dict]:
    try:
        ds_dict = load_dataset("FreedomIntelligence/evol-instruct-indonesian", streaming=args.streaming)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Skipping FreedomIntelligence/evol-instruct-indonesian: %s", exc
        )
        return iter(())
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            uid = row.get("id") or f"evol:{split_name}:{idx}"
            conv = row.get("conversations") or []
            messages = []
            for turn in conv:
                role = normalize_role(turn.get("from", "human"))
                content = (turn.get("value") or "").strip()
                if content:
                    messages.append({"role": role, "content": content})
            if not messages:
                continue
            output_text = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                "",
            )
            yield make_record(
                uid=f"evol:{uid}",
                source="FreedomIntelligence/evol-instruct-indonesian",
                task_type="chat",
                role_messages=messages,
                output_text=output_text,
                metadata={"split": split_name},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_global_mmlu(args: argparse.Namespace) -> Iterator[Dict]:
    ds_dict = load_dataset("CohereLabs/Global-MMLU", "id", streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            sample_id = row.get("sample_id") or f"globalmmlu:{split_name}:{idx}"
            question = row.get("question", "").strip()
            options = {
                "A": row.get("option_a", ""),
                "B": row.get("option_b", ""),
                "C": row.get("option_c", ""),
                "D": row.get("option_d", ""),
            }
            answer_key = str(row.get("answer", "")).strip()
            user_prompt_lines = [question]
            for letter in ("A", "B", "C", "D"):
                if options[letter]:
                    user_prompt_lines.append(f"{letter}. {options[letter]}")
            role_messages = [
                {"role": "user", "content": "\n".join(user_prompt_lines)},
                {
                    "role": "assistant",
                    "content": format_mcq_output(answer_key, options),
                },
            ]
            yield make_record(
                uid=f"globalmmlu:{sample_id}",
                source="CohereLabs/Global-MMLU",
                task_type="qa_mcq",
                role_messages=role_messages,
                output_text=format_mcq_output(answer_key, options),
                metadata={
                    "split": split_name,
                    "answer_key": answer_key,
                    "options_text": options,
                    "subject": row.get("subject"),
                    "category": row.get("subject_category"),
                },
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_indommlu(args: argparse.Namespace) -> Iterator[Dict]:
    ds_dict = load_dataset("indolem/IndoMMLU", streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            rid = row.get("id") or f"indommlu:{split_name}:{idx}"
            question = (row.get("soal") or "").strip()
            options_block = row.get("jawaban") or ""
            options_lines = [
                line for line in options_block.splitlines() if line.strip()
            ]
            options = {}
            for line in options_lines:
                if len(line) >= 3 and line[1] == ".":
                    letter = line[0].strip().upper()
                    options[letter] = line[3:].strip()
            answer_key = str(row.get("kunci", "")).strip()
            user_prompt_lines = [question] + [f"{k}. {v}" for k, v in options.items()]
            mcq_out = format_mcq_output(answer_key, options)
            yield make_record(
                uid=f"indommlu:{rid}",
                source="indolem/IndoMMLU",
                task_type="qa_mcq",
                role_messages=[
                    {"role": "user", "content": "\n".join(user_prompt_lines)},
                    {"role": "assistant", "content": mcq_out},
                ],
                output_text=mcq_out,
                metadata={
                    "split": split_name,
                    "answer_key": answer_key,
                    "options_text": options,
                    "subject": row.get("subject"),
                    "category": row.get("level"),
                },
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_blend(args: argparse.Namespace) -> Iterator[Dict]:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required for BLEnD download")
    path = hf_hub_download(
        repo_id="nayeon212/BLEnD",
        filename="data/annotations/Indonesia_data.json",
        repo_type="dataset",
    )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    for idx, (rid, row) in enumerate(data.items()):
        question = row.get("question", "").strip()
        annotations = row.get("annotations") or []
        best = pick_majority_annotation(annotations) or ""
        yield make_record(
            uid=f"blend:{rid}",
            source="nayeon212/BLEnD",
            task_type="qa_short",
            role_messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": best},
            ],
            output_text=best,
            metadata={
                "split": "indonesia",
                "all_annotations": annotations,
                "idks": row.get("idks"),
            },
        )
        if args.max_per_source and idx + 1 >= args.max_per_source:
            break


def load_indoqa(args: argparse.Namespace) -> Iterator[Dict]:
    train_url = "https://drive.google.com/uc?id=1ND893H5x2gaPRRMJVajQ4hgqpopHoD0u"
    val_url = "https://drive.google.com/uc?id=1mq_foV72riXb1KVBirJzTFZEe7oa8f4f"

    try:
        ds_dict = load_dataset(  # type: ignore
            "json",
            data_files={"train": train_url, "validation": val_url},
            streaming=args.streaming,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Falling back to manual download for indoqa: %s", exc)
        import requests

        def fetch_rows(url: str) -> List[Dict]:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return json.loads(resp.text)

        data = {
            "train": fetch_rows(train_url),
            "validation": fetch_rows(val_url),
        }
        ds_dict = {name: data[name] for name in data}

    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            rid = row.get("id") or f"indoqa:{split_name}:{idx}"
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            category = row.get("category") or ""
            ctx = row.get("context") or ""
            span_start = row.get("span_start")
            span_end = row.get("span_end")
            out_text = answer if category != "UNANSWERABLE" else "unanswerable"
            yield make_record(
                uid=f"indoqa:{rid}",
                source="SEACrowd/indoqa",
                task_type="qa_span",
                role_messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": out_text},
                ],
                output_text=out_text,
                input_payload={"context": ctx},
                metadata={
                    "split": split_name,
                    "category": category,
                    "span_start": span_start,
                    "span_end": span_end,
                },
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_facqa(args: argparse.Namespace) -> Iterator[Dict]:
    if pd is None:
        logging.warning("Skipping SEACrowd/facqa: pandas not installed")
        return iter(())
    url = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/facqa_qa-factoid-itb/train_preprocess.csv"
    df = pd.read_csv(url)
    for idx, row in df.iterrows():
        rid = row.get("index") if "index" in row else idx
        q_raw = row.get("question", "[]")
        p_raw = row.get("passage", "[]")
        lbl_raw = row.get("seq_label", "[]")
        try:
            q_tokens = ast.literal_eval(q_raw)
            p_tokens = ast.literal_eval(p_raw)
            labels = ast.literal_eval(lbl_raw)
        except Exception:
            q_tokens = []
            p_tokens = []
            labels = []
        question = " ".join(q_tokens).replace("  ", " ").strip()
        passage = " ".join(p_tokens).replace("  ", " ").strip()
        answer = extract_facqa_answer(p_tokens, labels)
        yield make_record(
            uid=f"facqa:{rid}",
            source="SEACrowd/facqa",
            task_type="qa_span",
            role_messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            output_text=answer,
            input_payload={"context": passage},
            metadata={"split": "train_preprocess"},
        )
        if args.max_per_source and idx + 1 >= args.max_per_source:
            break


def load_squad_id(args: argparse.Namespace) -> Iterator[Dict]:
    # Use local files provided
    for rec in iter_local_squad_files():
        yield rec
        # max_per_source not applied here because iter_local_squad_files yields both files fully


def load_reasoning(args: argparse.Namespace) -> Iterator[Dict]:
    try:
        ds_dict = load_dataset("ChavyvAkvar/indonesian-reasoning-Converted", streaming=args.streaming)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping indonesian-reasoning: %s", exc)
        return iter(())
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            uid = row.get("id") or f"reasoning:{split_name}:{idx}"
            messages = row.get("messages") or []
            normalized_messages = []
            reasoning_raw = None
            final_answer = ""
            for m in messages:
                role = normalize_role(m.get("role", "assistant"))
                content = (m.get("content") or "").strip()
                if role == "assistant":
                    reasoning_raw, final_answer = parse_reasoning_blocks(content)
                    normalized_messages.append({"role": role, "content": final_answer})
                else:
                    normalized_messages.append({"role": role, "content": content})
            if not normalized_messages:
                continue
            yield make_record(
                uid=f"reasoning:{uid}",
                source="izzulgod/indonesian-reasoning",
                task_type="reasoning",
                role_messages=normalized_messages,
                output_text=final_answer,
                metadata={"split": split_name, "reasoning_raw": reasoning_raw},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_math_olympiad(args: argparse.Namespace) -> Iterator[Dict]:
    ds_dict = load_dataset("hafidhsoekma/math-olympiad-indonesian-benchmark", streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            question = row.get("question", "").strip()
            answer = str(row.get("final_answer", "")).strip()
            uid = row.get("id") or f"math:{split_name}:{idx}"
            yield make_record(
                uid=f"math:{uid}",
                source="hafidhsoekma/math-olympiad-indonesian-benchmark",
                task_type="math",
                role_messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                output_text=answer,
                metadata={"split": split_name, "source_tag": row.get("source")},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_indonli(args: argparse.Namespace) -> Iterator[Dict]:
    url = "https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl"
    ds_dict = load_dataset("json", data_files={"train": url}, streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        for idx, row in enumerate(split):
            rid = row.get("id") or f"indonli:{split_name}:{idx}"
            premise = row.get("premise", "").strip()
            hypothesis = row.get("hypothesis", "").strip()
            label_val = row.get("label")
            label = NLI_LABELS.get(label_val, str(label_val))
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nApakah hipotesis benar menurut premis?"
            yield make_record(
                uid=f"indonli:{rid}",
                source="afaji/indonli",
                task_type="nli",
                role_messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": label},
                ],
                output_text=label,
                metadata={"split": split_name, "label_id": label_val},
            )
            if args.max_per_source and idx + 1 >= args.max_per_source:
                break


def load_deddy(args: argparse.Namespace) -> Iterator[Dict]:
    ds_dict = load_dataset("Deddy/Indonesia-dataset-2023", streaming=args.streaming)  # type: ignore
    for split_name, split in ds_dict.items():
        buffer: List[str] = []
        idx = 0
        for row in split:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            buffer.append(text)
            if len(buffer) == 2:
                question, answer = buffer
                uid = f"deddy:{split_name}:{idx}"
                yield make_record(
                    uid=uid,
                    source="Deddy/Indonesia-dataset-2023",
                    task_type="qa_short",
                    role_messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    output_text=answer,
                    metadata={"split": split_name},
                )
                buffer.clear()
                idx += 1
                if args.max_per_source and idx >= args.max_per_source:
                    break
        if args.max_per_source and idx >= args.max_per_source:
            break


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


LOADERS = {
    "oasst1": load_oasst1,
    "Ichsan2895/OASST_Top1_Indonesian": load_ichsan_top1,
    "izzulgod/indonesian-conversation": load_izzulgod_conversation,
    "FreedomIntelligence/evol-instruct-indonesian": load_evol_instruct,
    "CohereLabs/Global-MMLU": load_global_mmlu,
    "indolem/IndoMMLU": load_indommlu,
    "nayeon212/BLEnD": load_blend,
    "SEACrowd/indoqa": load_indoqa,
    "SEACrowd/facqa": load_facqa,
    "SEACrowd/squad_id": load_squad_id,
    "izzulgod/indonesian-reasoning": load_reasoning,
    "hafidhsoekma/math-olympiad-indonesian-benchmark": load_math_olympiad,
    "afaji/indonli": load_indonli,
    "Deddy/Indonesia-dataset-2023": load_deddy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare unified Indonesian datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-source JSONL files.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(DEFAULT_SOURCES),
        help="Subset of sources to process (defaults to all known sources).",
    )
    parser.add_argument(
        "--max-per-source", type=int, help="Optional cap per source for quick tests."
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use datasets streaming mode when available.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=2000,
        help="Log progress every N records per source (stderr).",
    )
    return parser.parse_args()


def process_source(source: str, args: argparse.Namespace, output_dir: Path) -> int:
    if source == "local_squad_files":
        ds = build_dataset(
            factory=iter_local_squad_files,
            log_every=args.log_every,
            label=source,
        )
        if ds.num_rows == 0:
            logging.warning("%s has zero rows; skipping save", source)
            return 0
        out_path = output_dir / "local_squad_files"
        ds.save_to_disk(out_path)
        logging.info("%s -> %s (%d records)", source, out_path, ds.num_rows)
        return ds.num_rows

    loader = LOADERS.get(source)
    if loader is None:
        logging.error("Unknown source: %s", source)
        return 0

    out_path = output_dir / source.replace("/", "_")
    ds = build_dataset(
        factory=lambda: loader(args),
        log_every=args.log_every,
        label=source,
    )
    if ds.num_rows == 0:
        logging.warning("%s has zero rows; skipping save", source)
        return 0
    ds.save_to_disk(out_path)
    logging.info("%s -> %s (%d records)", source, out_path, ds.num_rows)
    return ds.num_rows


def iter_local_squad_files() -> Iterator[Dict]:
    for path in SQUAD_LOCAL_FILES:
        source = f"local_squad:{path.stem}"
        for rec in iter_local_squad(path, source):
            yield rec


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if load_dataset is None:
        logging.error("The 'datasets' package is required. Please install it first.")
        sys.exit(1)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for source in args.sources:
        count = process_source(source, args, output_dir)
        manifest.append(
            {
                "source": source,
                "count": count,
                "path": str((output_dir / f"{source.replace('/', '_')}").resolve()),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
