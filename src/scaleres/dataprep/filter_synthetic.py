#!/usr/bin/env python
"""Filter a raw synthetic corpus. Every generated corpus must pass through this.

Generation on taco is deliberately unfiltered (F48 design note): GlotLID and
s_disc live here, and this programme has twice shipped a filter that later
turned out to be wrong (F35, F38), so raw rescoreable output is the safer
artifact. This is where the judgement gets applied.

Four gates, in cost order so the cheap ones run first:

  1. DEGENERATE TEXT. ~1% of generations are repetition loops that ran to the
     token cap -- tails like " a a a a a a a a ..." (F48). They are caught by
     the repetition heuristics in quality_utils, which reject 100% of capped
     docs and 1.3% of normal ones. This is the gate the operator asked to be
     sure of.
  2. EXACT DUPLICATES on normalised text.
  3. LANGUAGE ID via GlotLID, using the language's accept-list.
  4. s_disc, the discriminative language score, at a configurable threshold.

Each gate's removals are counted and reported separately, because "how much did
we lose and to what" is the number that tells you whether a corpus is healthy or
whether the generator drifted.

    python -m scaleres.dataprep.filter_synthetic \\
        --in dataset/synthetic_raw/ban_real2 --lang balinese \\
        --out dataset/synthetic_clean/ban_real2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))
REPORT = ROOT / "autoresearch/experiments/results/filter_synthetic.json"

WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFKC", t or "")).strip().lower()


def h(t: str) -> str:
    return hashlib.blake2b(t.encode(), digest_size=8).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--lang", required=True, choices=["balinese", "cirebonese"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--s-disc-min", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="stop once this many est. tokens are kept (0 = all)")
    ap.add_argument("--chars-per-token", type=float, default=3.99)
    ap.add_argument("--no-lid", action="store_true",
                    help="skip GlotLID (heuristics + s_disc only)")
    a = ap.parse_args()

    from scaleres.dataprep.quality_utils import evaluate_text, LANG_CONFIGS
    from s_disc import s_disc_score, SCORER_VERSION

    cfg = LANG_CONFIGS[a.lang]
    lid = None
    if not a.no_lid:
        from lid_utils import check_text
        lid = check_text

    files = sorted(Path(a.inp).glob("*.jsonl"))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clean.jsonl"

    seen: set[str] = set()
    n_in = kept = kept_chars = 0
    drop = Counter()
    heur_reason = Counter()
    lid_labels = Counter()
    sdisc_kept: list[float] = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for f in files:
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    drop["malformed_line"] += 1
                    continue
                n_in += 1
                text = (r.get("text") or "").strip()
                if not text:
                    drop["empty"] += 1
                    continue

                # 1. degenerate / repetition / malformed tokens
                ev = evaluate_text(text, cfg, lid=None)
                if not ev.ok:
                    drop["heuristic"] += 1
                    heur_reason[ev.reason] += 1
                    continue

                # 2. exact duplicate
                key = h(norm(text))
                if key in seen:
                    drop["duplicate"] += 1
                    continue
                seen.add(key)

                # 3. language id
                if lid is not None:
                    ok, reason, label, prob = lid(text, a.lang)
                    lid_labels[label or "none"] += 1
                    if not ok:
                        drop["lid"] += 1
                        continue

                # 4. discriminative language score
                sd = s_disc_score(text, a.lang)
                if sd < a.s_disc_min:
                    drop["s_disc"] += 1
                    continue

                fout.write(json.dumps(
                    {"text": text, "s_disc": round(sd, 4),
                     "src_i": r.get("src_i"), "latent": r.get("latent")},
                    ensure_ascii=False) + "\n")
                kept += 1
                kept_chars += len(text)
                sdisc_kept.append(sd)
                if a.max_tokens and kept_chars / a.chars_per_token >= a.max_tokens:
                    break
            if a.max_tokens and kept_chars / a.chars_per_token >= a.max_tokens:
                break

    est = kept_chars / a.chars_per_token
    sdisc_kept.sort()
    summary = {
        "input": a.inp, "output": str(out_path),
        "lang": a.lang, "scorer_version": SCORER_VERSION,
        "s_disc_min": a.s_disc_min, "lid": not a.no_lid,
        "docs_in": n_in, "docs_kept": kept,
        "keep_rate": round(kept / max(1, n_in), 4),
        "est_tokens_kept": int(est), "chars_kept": kept_chars,
        "dropped": dict(drop), "heuristic_reasons": dict(heur_reason),
        "lid_labels": dict(lid_labels.most_common(8)),
        "s_disc_median": round(sdisc_kept[len(sdisc_kept) // 2], 4) if sdisc_kept else None,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    prev = json.loads(REPORT.read_text()) if REPORT.exists() else []
    prev = [p for p in prev if p.get("input") != a.inp]
    prev.append(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
