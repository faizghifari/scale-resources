#!/usr/bin/env python
"""
build_mcqa_midtrain.py -- turn the unified Indonesian midtraining pool into MCQA
supervision in the EXACT format `scaleres.eval.eval_mcqa` scores.

WHY THE FORMAT IS THE WHOLE POINT
---------------------------------
FINDINGS 9: all 20 from-scratch checkpoints collapse onto a single answer letter
and score that letter's frequency in the key. Perplexity spans 218 -> 3081 across
those checkpoints with no relationship to MCQA at all. That is not a knowledge
failure, it is a format failure -- the models have never seen a task where the
token after "Pasaut: " depends on the preceding options.

`eval_mcqa.py` does not generate and parse. It builds

    <initial_prompt>\n\n[<context>\n\n]<question>\n<a>. <text>\n<b>. <text>\n\n<answer_prompt>

and compares the log-prob of the single label tokens `a`/`b`/`c` as the
continuation. So midtrain data has to supervise that exact continuation. The
existing Global-MMLU rows store the answer as `A. 4` -- uppercase, letter plus
text -- which does not supervise the bare lowercase token the eval reads. This
builder emits the eval's own string, byte for byte, via the same prompt shape.

TWO CHOICES THAT DIRECTLY TARGET THE COLLAPSE
---------------------------------------------
1. `--shuffle-options` permutes the option order per item, so the correct answer
   is not correlated with source-file position.
2. `--balance-labels` resamples the permutation so the gold label is uniform over
   the available letters. A model trained on a skewed key learns the skew; that is
   precisely the failure being fixed, so leaving it unbalanced would reproduce it.

CROSS-LINGUAL FORMAT TRANSFER
-----------------------------
The questions in the pool are Indonesian. Target-language MCQA needs translation,
which is blocked on the generator choice. But the answer-slot behaviour is a
FORMAT skill, and this builder wraps Indonesian questions in the TARGET language's
prompt frame (`Pilih pasaut ane pinih beneh!` ... `Pasaut: `). Whether that alone
lifts a checkpoint off the majority-label floor is a cheap, self-contained test --
and it lets Gate G0 run now instead of waiting on a translation pipeline. Treat a
positive result as format transfer, not as target-language competence.

Usage:
    python -m scaleres.dataprep.build_mcqa_midtrain \
        --lang bali --out dataset/midtraining/mcqa_fmt_bali
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "dataset/midtraining/midtraining_unified_full"

# Copied from scaleres.eval.eval_mcqa.LANG_PROMPTS. Kept as a literal rather than
# imported so a change there cannot silently desynchronise training from eval --
# the assertion in `check_matches_eval()` is what keeps them equal.
LANG_PROMPTS = {
    "cirebon": {"initial": "Pilih jawaban sing paling bener!", "answer": "Jawaban: "},
    "bali": {"initial": "Pilih pasaut ane pinih beneh!", "answer": "Pasaut: "},
}
DEFAULT_LABELS = "abcdefgh"


def build_prompt(question, labels, texts, initial, answer_prompt, context=""):
    """Byte-identical to eval_mcqa.build_prompt."""
    parts = []
    if initial:
        parts.append(initial.rstrip())
    if context.strip():
        parts.append(context.strip())
    qc = [question.rstrip()] if question else []
    for lab, txt in zip(labels, texts):
        qc.append(f"{lab}. {txt}")
    if qc:
        parts.append("\n".join(qc))
    if answer_prompt:
        parts.append(answer_prompt)
    return "\n\n".join(parts)


def check_matches_eval():
    """Guard: our prompt must equal the evaluator's on the same input."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from scaleres.eval.eval_mcqa import build_prompt as eval_build, _get_lang_prompts

    item = {"question": "Q?", "context": "",
            "choices": {"label": ["a", "b"], "text": ["one", "two"]}}
    for lang in LANG_PROMPTS:
        ini, ans = _get_lang_prompts(lang)
        theirs, _, _ = eval_build(item, ini, ans)
        ours = build_prompt("Q?", ["a", "b"], ["one", "two"], ini, ans)
        if theirs != ours:
            raise AssertionError(
                f"prompt drift for {lang}:\n  eval={theirs!r}\n  ours={ours!r}")
    return True


def iter_mcq_rows(sources):
    from datasets import load_from_disk
    for name in sources:
        p = POOL / name
        if not p.exists():
            print(f"  !! missing {name}")
            continue
        d = load_from_disk(str(p))
        if hasattr(d, "keys") and not hasattr(d, "column_names"):
            d = d[list(d.keys())[0]]
        n = 0
        for r in d:
            if r.get("task_type") != "qa_mcq":
                continue
            meta = r.get("metadata") or {}
            opts = meta.get("options_text") or {}
            key = (meta.get("answer_key") or "").strip().upper()
            q = ""
            for m in r.get("role_messages") or []:
                if m.get("role") == "user":
                    q = (m.get("content") or "").split("\n")[0].strip()
            if not (q and opts and key in opts):
                continue
            yield {"question": q, "options": opts, "answer_key": key, "source": name}
            n += 1
        print(f"  {name}: {n} usable qa_mcq rows")


def render(row, lang, rng, shuffle, balance):
    ini, ans = LANG_PROMPTS[lang]["initial"], LANG_PROMPTS[lang]["answer"]
    items = list(row["options"].items())          # [(A, text), ...]
    gold_text = row["options"][row["answer_key"]]
    if shuffle:
        rng.shuffle(items)
    if balance:
        # Move the gold option to a uniformly-chosen slot so the key is flat.
        texts = [t for _, t in items]
        texts.remove(gold_text)
        slot = rng.randrange(len(items))
        texts.insert(slot, gold_text)
    else:
        texts = [t for _, t in items]
    labels = list(DEFAULT_LABELS[: len(texts)])
    gold = labels[texts.index(gold_text)]
    prompt = build_prompt(row["question"], labels, texts, ini, ans)
    return {"text": prompt + gold, "gold": gold, "n_options": len(texts),
            "source": row["source"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANG_PROMPTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*",
                    default=["indolem_IndoMMLU", "CohereLabs_Global-MMLU"])
    ap.add_argument("--shuffle-options", action="store_true", default=True)
    ap.add_argument("--balance-labels", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()

    check_matches_eval()
    print("prompt format matches eval_mcqa.build_prompt exactly")

    rng = random.Random(a.seed)
    rows = [render(r, a.lang, rng, a.shuffle_options, a.balance_labels)
            for r in iter_mcq_rows(a.sources)]
    if a.max_items:
        rows = rows[: a.max_items]
    if not rows:
        raise SystemExit("no usable rows -- is IndoMMLU still empty? see P0.1")

    from collections import Counter
    dist = Counter(r["gold"] for r in rows)
    tot = sum(dist.values())
    print(f"\n{len(rows)} items | gold-label distribution:")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]:6d}  {dist[k] / tot:6.2%}")
    print(f"majority-label baseline on this set: {max(dist.values()) / tot:.4f}")

    from datasets import Dataset
    out = Path(a.out)
    Dataset.from_list(rows).save_to_disk(str(out))
    (out / "BUILD_INFO.json").write_text(json.dumps({
        "lang": a.lang, "sources": a.sources, "seed": a.seed,
        "shuffle_options": a.shuffle_options, "balance_labels": a.balance_labels,
        "n": len(rows), "gold_dist": dict(dist),
        "format": "byte-identical to scaleres.eval.eval_mcqa.build_prompt + gold label",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nSAMPLE\n" + "-" * 60 + f"\n{rows[0]['text']}\n" + "-" * 60)


if __name__ == "__main__":
    main()
