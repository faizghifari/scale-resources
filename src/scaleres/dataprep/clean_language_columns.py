#!/usr/bin/env python
"""Language-clean every Balinese/Cirebonese column in the repo (F52-F55).

The audit (research/audit_language_purity.py) showed that most of the
Cirebonese material in this repo is not Cirebonese, and that the shipped
synthetic corpus is Javanese throughout while the Balinese half of the same
pipeline is fine. This applies the gate and writes cleaned copies.

Gates, cheapest first:
  1. empty / too short
  2. exact duplicate on normalised target text
  3. degenerate text -- repetition loops, over-long tokens (quality_utils)
  4. LANGUAGE -- s_disc against the anchor midpoint

Rows are gated on the TARGET column but written WHOLE, so a parallel pair keeps
its source side and stays aligned.

An output is only written when enough survives to be worth having. Below
--min-keep the corpus is reported as unsalvageable and nothing is written --
a 6%-pure corpus does not become usable by taking its top 6%, it becomes a tiny
biased sample of whatever the scorer happens to like, and shipping that under
the old name is how F54 survived for months.

    python -m scaleres.dataprep.clean_language_columns            # all
    python -m scaleres.dataprep.clean_language_columns --lang balinese
"""
from __future__ import annotations

import argparse
import hashlib
import random
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))
REPORT = ROOT / "autoresearch/experiments/results/language_cleanup.json"

WS = re.compile(r"\s+")

# (path, lang, column, out). out=None -> <path>_langclean
TARGETS = [
    ("dataset/parallel/3lang/balinese_annotation-filter_bt-valid-pct_85", "balinese", "balinese", None),
    ("dataset/parallel/3lang/balinese_annotation-filter_valid-pct_50",    "balinese", "balinese", None),
    ("dataset/parallel/3lang/combined_705k_dedup_clean_filtered-id",      "balinese", "balinese", None),
    ("dataset/parallel/3lang/filtered_1k_short_nowiki",                   "balinese", "balinese", None),
    ("dataset/parallel/synthetic/raw",                                    "balinese", "balinese", None),
    ("dataset/parallel/synthetic/filtered_heuristic",                     "balinese", "balinese", None),
    ("dataset/parallel/synthetic_clean/clean_v1",                         "balinese", "balinese", None),
    ("dataset/parallel/2lang/id_cbn_127k_filtered",   "cirebonese", "translated_text", None),
    ("dataset/parallel/2lang/jv_cbn_24k",             "cirebonese", "translated_text", None),
    ("dataset/parallel/2lang/su_cbn_28k",             "cirebonese", "translated_text", None),
    ("dataset/parallel/3lang/cirebonese_annotation-filter_bt-valid-pct_80", "cirebonese", "cirebonese", None),
    ("dataset/parallel/3lang/cirebonese_annotation-filter_valid-pct_50",    "cirebonese", "cirebonese", None),
    ("dataset/parallel/3lang/combined_705k_dedup_clean_filtered-id",        "cirebonese", "cirebonese", None),
    ("dataset/parallel/3lang/filtered_1k_short_nowiki",                     "cirebonese", "cirebonese", None),
    ("dataset/parallel/synthetic/raw",                "cirebonese", "cirebonese", None),
    ("dataset/parallel/synthetic/filtered_heuristic", "cirebonese", "cirebonese", None),
    ("dataset/parallel/synthetic_clean/clean_v1",     "cirebonese", "cirebonese", None),
]
MIN_CHARS = 40


def norm(t):
    return WS.sub(" ", unicodedata.normalize("NFKC", t or "")).strip().lower()


def h(t):
    return hashlib.blake2b(t.encode(), digest_size=8).hexdigest()


def threshold(lang, anchors):
    anc = anchors[lang]
    rivals = {k: v for k, v in anc.items()
              if k not in ("real_target", "real_balinese")}
    return (anc["real_target"] + max(rivals.values())) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["balinese", "cirebonese"])
    ap.add_argument("--min-keep", type=float, default=0.35,
                    help="below this keep rate, report and write nothing")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = whole dataset; otherwise a RANDOM sample of this "
                         "size. These corpora are ordered and their heads are "
                         "not representative -- the first 4,000 rows of "
                         "synthetic_clean/clean_v1 are 33.7% pure against 90.5% "
                         "for a random 4,000 -- so a prefix would mislead.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--salvage", action="store_true",
                    help="also write sub-threshold corpora, under a _sdisc_salvage "
                         "name. These are NOT clean corpora: keeping the top 7%% of "
                         "a 7%%-pure corpus yields whatever the scorer likes most, "
                         "which is a biased sample, not a representative one. Named "
                         "so it cannot be mistaken for cleaned data.")
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from s_disc import s_disc_score, ANCHORS, SCORER_VERSION
    from scaleres.dataprep.quality_utils import evaluate_text, LANG_CONFIGS

    print(f"scorer {SCORER_VERSION}\n")
    hdr = (f"{'dataset':56s} {'col':16s} {'in':>9} {'kept':>9} {'keep%':>7} "
           f"{'median':>8}  status")
    print(hdr); print("-" * len(hdr))

    results = []
    for path, lang, col, out in TARGETS:
        if a.lang and lang != a.lang:
            continue
        p = ROOT / path
        if not p.exists():
            print(f"{path:56s} {col:16s} MISSING")
            continue
        thr = threshold(lang, ANCHORS)
        cfg = LANG_CONFIGS[lang]

        d = load_from_disk(str(p))
        if hasattr(d, "keys") and not hasattr(d, "column_names"):
            d = d[list(d.keys())[0]]
        if col not in d.column_names:
            print(f"{path:56s} {col:16s} NO SUCH COLUMN")
            continue
        if a.limit and a.limit < len(d):
            rng = random.Random(a.seed)
            sel = sorted(rng.sample(range(len(d)), a.limit))
        else:
            sel = list(range(len(d)))
        n = len(sel)
        view = d.select(sel)

        seen, keep_idx, kept_scores = set(), [], []
        drop = Counter()
        texts = view[col]
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if len(t) < MIN_CHARS:
                drop["short"] += 1
                continue
            k = h(norm(t))
            if k in seen:
                drop["duplicate"] += 1
                continue
            seen.add(k)
            if not evaluate_text(t, cfg, lid=None).ok:
                drop["degenerate"] += 1
                continue
            s = s_disc_score(t, lang)
            if s < thr:
                drop["language"] += 1
                continue
            keep_idx.append(i)
            kept_scores.append(s)

        rate = len(keep_idx) / max(1, n)
        med = (sorted(kept_scores)[len(kept_scores) // 2]
               if kept_scores else float("nan"))
        if rate < a.min_keep:
            if a.salvage and keep_idx:
                outp = f"{path}_sdisc_salvage_{lang[:3]}"
                view.select(keep_idx).save_to_disk(str(ROOT / outp))
                status = f"BIASED SALVAGE -> {outp}"
            else:
                status = "UNSALVAGEABLE -- not written"
                outp = None
        elif a.dry_run:
            status = "ok (dry-run)"
            outp = None
        else:
            suffix = "_langclean" if col in ("text", "translated_text") \
                else f"_langclean_{lang[:3]}"
            outp = (out or f"{path}{suffix}")
            view.select(keep_idx).save_to_disk(str(ROOT / outp))
            status = f"wrote {outp}"

        print(f"{path:56s} {col:16s} {n:>9,} {len(keep_idx):>9,} "
              f"{100*rate:>6.1f}% {med:>+8.3f}  {status}")
        results.append({
            "path": path, "lang": lang, "column": col, "threshold": round(thr, 4),
            "rows_in": n, "rows_total": len(d), "sampled": bool(a.limit and a.limit < len(d)),
            "rows_kept": len(keep_idx), "keep_rate": round(rate, 4),
            "kept_s_disc_median": round(med, 4) if kept_scores else None,
            "dropped": dict(drop), "output": outp, "status": status,
        })

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"scorer_version": SCORER_VERSION, "min_keep": a.min_keep,
         "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
