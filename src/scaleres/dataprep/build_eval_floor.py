#!/usr/bin/env python
"""S2.4 -- build a verified-unseen OOD eval set per expansion language.

OOD perplexity is one of only two surviving outcome measures in this programme
(F3/F9), so the eval set has to be provably unseen. It is not a formality: the
existing Balinese set went from 998 candidates to 523 after decontamination --
48% of it was already in training.

Sources, both held out from the pretraining pile by construction:
  * NusaX-senti  -- human-written sentences, language-verified (min/ace/bug)
  * FLORES+      -- professionally translated parallel text (min/ace/bug/ban),
                    which also makes the eval comparable ACROSS languages,
                    since every language gets the same 2,009 source sentences.

Decontamination is exhaustive, not sampled. The n-gram index is built from the
EVAL side (a few thousand short items) and the training corpus is streamed
against it, rather than the other way round -- indexing 68 MB of Minangkabau
would cost gigabytes for no gain.

    python -m scaleres.dataprep.build_eval_floor --langs min ace bug
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "dataset/clean"
RAW = ROOT / "dataset/raw"
OUTDIR = ROOT / "dataset/eval"
REPORT = ROOT / "autoresearch/experiments/results/S2_4_eval_floor.json"

NGRAM = 10           # matches the 10-gram check used for the Balinese set
MIN_CHARS = 30
WS = re.compile(r"\s+")

FLORES = "openlanguagedata/flores_plus"
FLORES_CFG = {"min": "min_Latn", "ace": "ace_Latn", "bug": "bug_Latn",
              "ban": "ban_Latn"}


def norm(t: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFKC", t or "")).strip().lower()


def h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def ngrams(text: str, n=NGRAM):
    w = norm(text).split()
    for i in range(max(0, len(w) - n + 1)):
        yield h(" ".join(w[i:i + n]))


def load_candidates(lang: str) -> list[dict]:
    """Eval candidates from every held-out source available for this language."""
    from datasets import load_dataset
    out = []

    nusax = RAW / lang / "task" / "indonlp__NusaX-senti.jsonl"
    if nusax.exists():
        for line in open(nusax, encoding="utf-8"):
            r = json.loads(line)
            if len(r.get("text") or "") >= MIN_CHARS:
                out.append({"text": r["text"], "source": "NusaX-senti"})

    cfg = FLORES_CFG.get(lang)
    if cfg:
        try:
            ds = load_dataset(FLORES, cfg)
            for split in ds:
                for r in ds[split]:
                    t = r.get("text") or ""
                    if len(t) >= MIN_CHARS:
                        out.append({"text": t, "source": f"FLORES+/{split}",
                                    "flores_id": r.get("id")})
        except Exception as ex:
            print(f"    FLORES+ {cfg}: {type(ex).__name__}: {str(ex)[:120]}")
    return out


def decontaminate(cands: list[dict], lang: str) -> tuple[list[dict], dict]:
    """Drop any candidate sharing a 10-gram with the training corpus."""
    index: dict[int, set[int]] = defaultdict(set)
    skipped_short = 0
    for i, c in enumerate(cands):
        gs = list(ngrams(c["text"]))
        if not gs:
            skipped_short += 1          # shorter than one n-gram; cannot check
            continue
        for g in gs:
            index[g].add(i)

    train = CLEAN / lang / "mono.jsonl"
    hit: set[int] = set()
    n_train = 0
    if train.exists():
        with open(train, encoding="utf-8") as f:
            for line in f:
                n_train += 1
                for g in ngrams(json.loads(line).get("text") or ""):
                    if g in index:
                        hit |= index[g]
    keep = [c for i, c in enumerate(cands) if i not in hit]
    stats = {
        "candidates": len(cands),
        "training_docs_scanned": n_train,
        "removed_contaminated": len(hit),
        "unverifiable_too_short": skipped_short,
        "kept": len(keep),
        "contamination_rate": round(len(hit) / max(1, len(cands)), 4),
    }
    return keep, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["min", "ace", "bug"])
    a = ap.parse_args()
    from datasets import Dataset

    results = []
    for lang in a.langs:
        print(f"\n=== {lang} ===", flush=True)
        cands = load_candidates(lang)
        by_src = defaultdict(int)
        for c in cands:
            by_src[c["source"].split("/")[0]] += 1
        print(f"  candidates: {len(cands):,}  {dict(by_src)}")
        if not cands:
            results.append({"lang": lang, "error": "no candidates"})
            continue

        keep, stats = decontaminate(cands, lang)
        print(f"  scanned {stats['training_docs_scanned']:,} training docs")
        print(f"  removed {stats['removed_contaminated']:,} contaminated "
              f"({stats['contamination_rate']:.1%}) -> {stats['kept']:,} kept")

        out = OUTDIR / lang / "ood_clean"
        out.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list([{"text": c["text"], "source": c["source"]} for c in keep]
                          ).save_to_disk(str(out))
        kept_by_src = defaultdict(int)
        for c in keep:
            kept_by_src[c["source"]] += 1
        stats.update(lang=lang, output=str(out.relative_to(ROOT)),
                     kept_by_source=dict(kept_by_src))
        results.append(stats)
        print(f"  wrote {out.relative_to(ROOT)}  {dict(kept_by_src)}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "exp": "S2.4_eval_floor",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ngram": NGRAM,
        "note": ("Verified-unseen OOD sets. Decontaminated exhaustively against "
                 "dataset/clean/<lang>/mono.jsonl by 10-gram overlap. Items too "
                 "short to form one 10-gram are reported separately -- they are "
                 "not verified, merely unchecked."),
        "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
