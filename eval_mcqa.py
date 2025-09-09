#!/usr/bin/env python
"""
eval_mcqa.py - Multiple Choice QA evaluator using direct label log-prob scoring.

Workflow per question:
1. Build a prompt:
    [lang_initial_prompt]\n[optional context]\nQuestion: <question>\nChoices:\n<a>. <text_for_a>\n<b>. <text_for_b> ...\n[lang_answer_prompt]
2. Score each choice label (e.g. 'a','b','c') as the continuation tokens immediately
   following the prompt. The label with highest total log-prob is selected.

Rationale:
Avoids generation + parsing; directly measures model preference among provided labels.

Assumptions:
- JSON file is a list of objects each containing: context (str), question (str),
  choices: {"label": [...], "text": [...]}, answer (gold label), question_id (int).
- Labels are short (typically single characters). Multi-token labels are supported.

Outputs:
- Directory: eval_mcqa/<model_name>/<run_name>/
  * details.jsonl  (one JSON per question with per-label logprobs and prediction)
  * summary.json   (aggregate accuracy overall, by category, by grade)

Language presets (selected via --lang):
    cirebon: initial="Pilih jawaban sing paling bener!" answer="Jawaban: "
    bali:    initial="Pilih pasaut ane pinih beneh!"   answer="Pasaut: "

Supports optional WandB logging (overall accuracy + per-grade/category metrics).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
import time
import math
from statistics import mean, median
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MCQA via label log-probs")
    p.add_argument("--model_path_or_id", required=True)
    p.add_argument("--tokenizer_path_or_id", default=None)
    p.add_argument("--data_path", required=True, help="Path to MCQA JSON list file")
    p.add_argument(
        "--lang",
        required=True,
        choices=["cirebon", "bali"],
        help="Language preset for initial + answer prompts",
    )
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="(Reserved) - currently scoring question-by-question",
    )
    p.add_argument("--report_to", choices=["none", "wandb"], default="wandb")
    p.add_argument("--wandb_project", default="EvalMCQA")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument(
        "--save_logits",
        action="store_true",
        help="If set, stores per-token logprobs for each label (may increase file size)",
    )
    p.add_argument(
        "--print_prompts",
        type=int,
        default=0,
        help="Print first N constructed prompts (with token + character counts) before evaluation.",
    )
    p.add_argument(
        "--print_and_exit",
        action="store_true",
        help="After printing/saving prompts (if requested), exit without scoring questions.",
    )
    return p.parse_args()


LANG_PROMPTS = {
    "cirebon": {
        "initial": "Pilih jawaban sing paling bener!",
        "answer": "Jawaban: ",  # keep trailing space
    },
    "bali": {
        "initial": "Pilih pasaut ane pinih beneh!",
        "answer": "Pasaut: ",  # keep trailing space
    },
}


def _get_lang_prompts(lang: str) -> Tuple[str, str]:
    cfg = LANG_PROMPTS[lang]
    # initial trimmed right, answer preserved (including trailing space) to separate label
    return cfg["initial"].rstrip(), cfg["answer"]


def _model_name_from_path_or_id(src: str) -> str:
    if os.path.isdir(src):
        name = os.path.basename(os.path.normpath(src))
    else:
        name = src.split("/")[-1]
    return name or "model"


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("MCQA data must be a list of question objects")
    return cast(List[Dict[str, Any]], data)


def build_prompt(
    item: Dict[str, Any],
    initial_prompt: str,
    answer_prompt: str,
) -> Tuple[str, List[str], List[str]]:
    context = (item.get("context") or "").strip()
    question = (item.get("question") or "").strip()
    choices = item.get("choices") or {}
    labels = list(choices.get("label") or [])
    texts = list(choices.get("text") or [])
    if len(labels) != len(texts):
        raise ValueError("Mismatch labels vs texts length")

    parts: List[str] = []
    if initial_prompt:
        parts.append(initial_prompt.rstrip())
    if context:
        parts.append(context.rstrip())
    # Question + choices (single newlines inside this block)
    qc_lines: List[str] = [question.rstrip()] if question else []
    for lab, txt in zip(labels, texts):
        qc_lines.append(f"{lab}. {txt}")
    if qc_lines:
        parts.append("\n".join(qc_lines))
    if answer_prompt:
        # DO NOT strip trailing space of answer_prompt (used to separate label token)
        parts.append(answer_prompt)
    prompt = "\n\n".join(parts)
    return prompt, labels, texts


def tokenize_labels(
    tokenizer: Any, labels: List[str]
) -> Tuple[Dict[str, List[int]], bool]:
    label_id_map: Dict[str, List[int]] = {}
    all_single = True
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(lab, add_special_tokens=False)
        label_id_map[lab] = ids
        if len(ids) != 1:
            all_single = False
    return label_id_map, all_single


def score_labels_fallback(
    model: Any,
    prompt_ids: List[int],
    label_id_map: Dict[str, List[int]],
    device: torch.device,
    save_logits: bool,
) -> Dict[str, Any]:
    # Original per-label extension method (handles multi-token labels)
    results: Dict[str, Any] = {}
    prompt_len = len(prompt_ids)
    for lab, lab_ids in label_id_map.items():
        input_ids = torch.tensor(
            prompt_ids + lab_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            label_logprob = 0.0
            per_token: List[float] = []
            for i, tok in enumerate(lab_ids):
                pos = prompt_len + i
                prev = pos - 1
                token_logits = logits[0, prev]
                log_probs = F.log_softmax(token_logits, dim=-1)
                lp = float(log_probs[tok].item())
                label_logprob += lp
                per_token.append(lp)
        results[lab] = {
            "logprob": label_logprob,
            **({"per_token_logprobs": per_token} if save_logits else {}),
            "length": len(lab_ids),
        }
    return results


def score_labels_fast(
    model: Any,
    prompt_ids: List[int],
    label_id_map: Dict[str, List[int]],
    device: torch.device,
    save_logits: bool,
) -> Dict[str, Any]:
    # Single forward pass, all labels must be single-token
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
        logits_last = outputs.logits[0, -1]  # [vocab]
        log_probs = F.log_softmax(logits_last, dim=-1)
    results: Dict[str, Any] = {}
    for lab, ids in label_id_map.items():
        tok_id = ids[0]
        lp = float(log_probs[tok_id].item())
        results[lab] = {
            "logprob": lp,
            **({"per_token_logprobs": [lp]} if save_logits else {}),
            "length": 1,
        }
    return results


def aggregate_summary(
    rows: List[Dict[str, Any]], meta: Dict[str, Any]
) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    overall_acc = correct / total if total else 0.0
    by_category: Dict[str, List[bool]] = defaultdict(list)
    by_grade: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        cats = r.get("category") or []
        for c in cats:
            by_category[c].append(r["correct"])
        grade = str(r.get("grade"))
        by_grade[grade].append(r["correct"])
    cat_metrics = {c: sum(v) / len(v) if v else 0.0 for c, v in by_category.items()}
    grade_metrics = {g: sum(v) / len(v) if v else 0.0 for g, v in by_grade.items()}
    summary = {
        "total_questions": total,
        "overall_accuracy": overall_acc,
        "per_category_accuracy": cat_metrics,
        "per_grade_accuracy": grade_metrics,
    }
    summary.update(meta)
    return summary


def validate_question(item: Dict[str, Any]) -> Optional[str]:
    choices = item.get("choices") or {}
    labels = list(choices.get("label") or [])
    texts = list(choices.get("text") or [])
    if not labels:
        return "no_labels"
    if len(labels) != len(texts):
        return "mismatched_label_text"
    if len(set(labels)) != len(labels):
        return "duplicate_labels"
    gold = item.get("answer")
    if gold not in labels:
        return "gold_not_in_labels"
    return None


def prepare_model_and_tokenizer(args, device: torch.device):
    tok_src = args.tokenizer_path_or_id or args.model_path_or_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False)
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        try:
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore
        except Exception:
            pass
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_id, torch_dtype=dtype
    )
    model.to(device)  # type: ignore
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()

    initial_prompt, answer_prompt = _get_lang_prompts(args.lang)

    data = load_data(args.data_path)
    if args.max_questions is not None:
        data = data[: args.max_questions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = prepare_model_and_tokenizer(args, device)
    model = cast(Any, model)

    model_name = _model_name_from_path_or_id(args.model_path_or_id)
    run_name = args.wandb_run_name or model_name

    # WandB (optional)
    use_wandb = args.report_to == "wandb"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore

            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model_path_or_id,
                    "data_path": args.data_path,
                    "max_questions": args.max_questions,
                    "lang": args.lang,
                },
            )
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")
            use_wandb = False

    out_root = os.path.join("eval_mcqa", model_name, run_name)
    os.makedirs(out_root, exist_ok=True)
    details_path = os.path.join(out_root, "details.jsonl")

    rows: List[Dict[str, Any]] = []
    skipped_reasons: Dict[str, int] = defaultdict(int)
    latencies_ms: List[float] = []

    # Prompt inspection (optional) before scoring
    if args.print_prompts > 0:
        to_print = args.print_prompts
        for idx, item in enumerate(data):
            if args.max_questions is not None and idx >= args.max_questions:
                break
            prompt, labels, choice_texts = build_prompt(
                item, initial_prompt, answer_prompt
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            qid = item.get("question_id", idx)
            if to_print > 0:
                print("=" * 80)
                print(f"Prompt #{args.print_prompts - to_print + 1} question_id={qid}")
                print(prompt)
                print(
                    f"[META] chars={len(prompt)} tokens={len(token_ids)} labels={labels}"
                )
                to_print -= 1
        if args.print_and_exit:
            print("Exiting after prompt inspection (print_and_exit).")
            return
    max_ctx = getattr(model.config, "max_position_embeddings", None)

    with open(details_path, "w", encoding="utf-8") as outf:
        for idx, item in enumerate(data):
            reason = validate_question(item)
            if reason is not None:
                skipped_reasons[reason] += 1
                continue
            prompt, labels, _ = build_prompt(item, initial_prompt, answer_prompt)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not prompt_ids:
                skipped_reasons["empty_prompt"] += 1
                continue
            if max_ctx is not None and len(prompt_ids) >= max_ctx:
                # Truncate from the left (keep last max_ctx-1 tokens to predict label)
                prompt_ids = prompt_ids[-(max_ctx - 1) :]
            label_id_map, all_single = tokenize_labels(tokenizer, labels)
            start_t = time.perf_counter()
            try:
                if all_single:
                    label_scores = score_labels_fast(
                        model, prompt_ids, label_id_map, device, args.save_logits
                    )
                else:
                    label_scores = score_labels_fallback(
                        model, prompt_ids, label_id_map, device, args.save_logits
                    )
            except RuntimeError as e:
                skipped_reasons["runtime_error"] += 1
                continue
            lat_ms = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(lat_ms)
            pred_label = (
                max(labels, key=lambda l: label_scores[l]["logprob"])
                if labels
                else None
            )
            gold = item.get("answer")
            correct = pred_label == gold
            row = {
                "question_id": item.get("question_id", idx),
                "gold": gold,
                "pred": pred_label,
                "correct": correct,
                "labels": labels,
                "label_scores": {l: label_scores[l]["logprob"] for l in labels},
                **(
                    {
                        "label_token_logprobs": {
                            l: label_scores[l].get("per_token_logprobs") for l in labels
                        }
                    }
                    if args.save_logits
                    else {}
                ),
                "category": item.get("category"),
                "grade": item.get("grade"),
                "latency_ms": round(lat_ms, 3),
                "all_labels_single_token": all_single,
            }
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (len(rows) + 1) % 50 == 0:
                outf.flush()
            rows.append(row)

    meta = {
        "skipped_total": sum(skipped_reasons.values()),
        "skipped_breakdown": skipped_reasons,
        "avg_latency_ms": mean(latencies_ms) if latencies_ms else None,
        "median_latency_ms": median(latencies_ms) if latencies_ms else None,
        "p95_latency_ms": (
            sorted(latencies_ms)[int(math.ceil(0.95 * len(latencies_ms))) - 1]
            if latencies_ms
            else None
        ),
        "model_max_position_embeddings": getattr(
            model.config, "max_position_embeddings", None
        ),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    summary = aggregate_summary(rows, meta)
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved MCQA evaluation to {out_root}")
    print(json.dumps(summary, indent=2))

    if use_wandb and wandb is not None:
        log_payload = {"overall_accuracy": summary["overall_accuracy"]}
        # Flatten some metrics for convenience
        for k, v in summary["per_category_accuracy"].items():
            log_payload[f"cat_acc/{k}"] = v
        for k, v in summary["per_grade_accuracy"].items():
            log_payload[f"grade_acc/{k}"] = v
        try:
            wandb.log(log_payload)
            wandb.finish()
        except Exception as e:
            print(f"[WARN] wandb logging failed: {e}")


if __name__ == "__main__":
    main()
