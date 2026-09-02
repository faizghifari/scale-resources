#!/usr/bin/env python
"""Translate Indonesian MCQA content into a target language, keeping the frame.

WHY THIS AND NOT MORE ARM A. G0 established that Indonesian task data teaches
the answer slot and supplies no knowledge (F9), and that the missing ingredient
is TARGET-language task data. Meanwhile the programme already holds 316M
synthetic Balinese and 108M synthetic Cirebonese tokens against 14.1M and 4.7M
real ones, so more free-generated prose is the low-yield axis (F1). This is the
cheap thing that unblocks a gate.

SAFETY OF THE DESIGN. The model never sees the gold label and never emits the
option letters. It is asked for strict JSON containing only the question stem
and the option TEXTS; the letters, their order, the surrounding prompt frame
and the answer line are all reassembled deterministically afterwards. A model
that silently reorders options would otherwise corrupt every label in the set,
and that corruption would be invisible until a training run failed.

Every translation is scored with s_disc, so the output can be filtered on the
one language instrument that has a measured error rate (F22). s_disc measures
LANGUAGE, not translation adequacy -- per F4 no instrument here has been shown
to predict downstream quality, and this one does not either.

    python -m scaleres.generation.translate_mcqa --lang balinese --n 300
    python -m scaleres.generation.translate_mcqa --lang cirebonese --n 0   # all
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))

SRC = {"balinese": ROOT / "dataset/midtraining/mcqa_fmt_bali",
       "cirebonese": ROOT / "dataset/midtraining/mcqa_fmt_cirebon"}
OUTDIR = ROOT / "dataset/midtraining/mcqa_translated"
REPORT = ROOT / "autoresearch/experiments/results/translate_mcqa.json"

LANG_NAME = {"balinese": "Balinese (basa Bali)",
             "cirebonese": "Cirebonese (basa Cirebon / Jawa Cirebon)"}

SYSTEM = (
    "You are a careful translator into {lang}. You translate Indonesian exam "
    "questions into natural {lang} for schoolchildren. You never answer the "
    "question, never add commentary, and never change the number or order of "
    "the answer options. You reply with JSON only."
)

USER = """Translate this Indonesian multiple-choice question into {lang}.

Rules:
- Translate the question stem and every option text into {lang}.
- Keep proper nouns, numbers, dates and formulas exactly as they are.
- Do NOT translate into Indonesian. Do NOT answer the question.
- Return the options in the SAME ORDER you were given.

Question:
{question}

Options (in order):
{options}

Reply with JSON only, in exactly this shape and nothing else:
{{"question": "<translated question>", "options": [{opt_slots}]}}"""


def parse_item(text: str) -> dict | None:
    """Split the rendered prompt back into header / question / options / tail."""
    lines = text.split("\n")
    if not lines:
        return None
    header = lines[0]
    body = "\n".join(lines[1:])
    m = list(re.finditer(r"(?m)^([a-e])\.\s(.*)$", body))
    if len(m) < 2:
        return None
    question = body[:m[0].start()].strip()
    options = [(x.group(1), x.group(2).strip()) for x in m]
    tail_start = m[-1].end()
    tail = body[tail_start:].strip()
    if not question:
        return None
    return {"header": header, "question": question,
            "options": options, "tail": tail}


def render(parsed: dict, question: str, opt_texts: list[str], gold: str) -> str:
    lines = [parsed["header"], "", question, ""]
    for (letter, _), new in zip(parsed["options"], opt_texts):
        lines.append(f"{letter}. {new}")
    answer_prefix = parsed["tail"].split(":")[0] if ":" in parsed["tail"] else "Pasaut"
    lines.append("")
    lines.append(f"{answer_prefix}: {gold}")
    return "\n".join(lines)


def translate_one(client, model, lang, row, idx):
    import llm_client as lc
    from s_disc import s_disc_score

    parsed = parse_item(row["text"])
    if not parsed:
        return {"i": idx, "status": "unparseable"}
    opts = parsed["options"]
    opt_block = "\n".join(f"{i+1}. {t}" for i, (_, t) in enumerate(opts))
    slots = ", ".join(f'"<option {i+1}>"' for i in range(len(opts)))
    try:
        r = lc.chat(
            client,
            SYSTEM.format(lang=LANG_NAME[lang]),
            USER.format(lang=LANG_NAME[lang], question=parsed["question"],
                        options=opt_block, opt_slots=slots),
            seed=20260902 + idx, model=model, max_tokens=1200, temperature=0.3)
    except Exception as ex:
        return {"i": idx, "status": "error", "error": f"{type(ex).__name__}: {ex}"[:160]}

    raw = (r["content"] or "").strip()
    mjson = re.search(r"\{.*\}", raw, re.S)
    if not mjson:
        return {"i": idx, "status": "no_json", "raw": raw[:160]}
    try:
        obj = json.loads(mjson.group(0))
    except json.JSONDecodeError:
        return {"i": idx, "status": "bad_json", "raw": raw[:160]}

    q = (obj.get("question") or "").strip()
    new_opts = [str(x).strip() for x in (obj.get("options") or [])]
    # Structural guards. A wrong option count means the label mapping is no
    # longer trustworthy, so the item is dropped rather than repaired.
    if not q or len(new_opts) != len(opts) or any(not o for o in new_opts):
        return {"i": idx, "status": "shape_mismatch",
                "want": len(opts), "got": len(new_opts)}
    # The model sometimes emits the literal strings "None"/"null" for options it
    # declined to translate. Those are non-empty, so the emptiness check above
    # passes them straight through into what looks like a valid item.
    if any(o.lower() in ("none", "null", "n/a", "-") for o in new_opts):
        return {"i": idx, "status": "null_option"}
    # An untranslated item is worse than a missing one: it trains the model that
    # the target language IS Indonesian. Require the output to differ from the
    # input for the stem and at least half the options.
    unchanged = sum(1 for (_, old), new in zip(opts, new_opts)
                    if old.strip().lower() == new.lower())
    if q.lower() == parsed["question"].lower() or unchanged > len(opts) // 2:
        return {"i": idx, "status": "untranslated"}

    text = render(parsed, q, new_opts, row["gold"])
    return {"i": idx, "status": "ok", "text": text, "gold": row["gold"],
            "n_options": row["n_options"], "source": row["source"],
            "s_disc": round(s_disc_score(q + " " + " ".join(new_opts), lang), 4),
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "chars": len(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["balinese", "cirebonese"])
    ap.add_argument("--base-url", default="http://143.248.188.121:10081/v1")
    ap.add_argument("--model", default="m")
    ap.add_argument("--n", type=int, default=300, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default="pilot")
    a = ap.parse_args()

    import llm_client as lc
    from datasets import load_from_disk
    from s_disc import ANCHORS
    lc.LOCAL_API_KEY = "not-needed"
    client = lc.get_client(a.base_url)

    ds = load_from_disk(str(SRC[a.lang]))
    n = len(ds) if a.n == 0 else min(a.n, len(ds))
    step = max(1, len(ds) // n)
    idxs = list(range(0, len(ds), step))[:n]
    print(f"{a.lang}: translating {len(idxs):,} of {len(ds):,} items "
          f"via {a.base_url}", flush=True)

    t0 = time.time()
    out = []
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(translate_one, client, a.model, a.lang, ds[i], i)
                for i in idxs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            out.append(f.result())
            if k % 50 == 0:
                ok = sum(1 for r in out if r["status"] == "ok")
                print(f"  {k}/{len(idxs)}  ok={ok}  "
                      f"{k/(time.time()-t0)*60:.1f} items/min", flush=True)
    wall = time.time() - t0

    ok = [r for r in out if r["status"] == "ok"]
    from collections import Counter
    status = Counter(r["status"] for r in out)
    anchor = ANCHORS[a.lang]["real_target"]
    sd = sorted(r["s_disc"] for r in ok)
    med = sd[len(sd) // 2] if sd else 0.0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{a.lang}__{a.label}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps({k: r[k] for k in
                                ("text", "gold", "n_options", "source", "s_disc")},
                               ensure_ascii=False) + "\n")

    summary = {
        "lang": a.lang, "label": a.label, "model": a.model,
        "requested": len(idxs), "ok": len(ok),
        "status": dict(status), "wall_seconds": round(wall, 1),
        "items_per_min": round(len(idxs) / wall * 60, 1),
        "s_disc_median": round(med, 4),
        "s_disc_pass_rate": round(sum(1 for x in sd if x >= 0.0) / max(1, len(sd)), 4),
        "pct_at_real_anchor": round(sum(1 for x in sd if x >= anchor) / max(1, len(sd)), 4),
        "real_target_anchor": anchor,
        "output": str(path.relative_to(ROOT)),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print("\n=== " + a.lang + " ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    prev = json.loads(REPORT.read_text()) if REPORT.exists() else []
    prev = [p for p in prev if not (p["lang"] == a.lang and p["label"] == a.label)]
    prev.append(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
