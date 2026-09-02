#!/usr/bin/env python
"""S2.5 -- pull NusaX / NusaWrites data straight from GitHub.

The Hub route is broken for this family: every canonical `indonlp/` and
`SEACrowd/` repo is script-based and `datasets` 4.x refuses to load it (F32),
and the parquet mirrors cover only part of it. The GitHub repos hold the same
data as plain CSV, so they are both more complete and more stable.

Three things worth having, none of which the mirrors provide in full:

  * NusaX `datasets/lexicon/*.csv` -- BILINGUAL LEXICONS for acehnese,
    balinese, buginese, minangkabau and others. This is Arm C scaffolding
    material; the programme currently has hand-built lexicons for Balinese and
    Cirebonese only.
  * NusaWrites `nusa_kalimat-mt-*` -- sentence-level parallel translation,
    which is the axis-A translation ingredient for the new languages.
  * NusaWrites `nusa_alinea-paragraph-*` -- human-written PARAGRAPHS, i.e. real
    prose rather than task items, which is scarce for all five targets.

    python -m scaleres.dataprep.pull_nusa_github
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "autoresearch/experiments/results/S2_5_nusa_github.json"
RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"
TREE = "https://api.github.com/repos/{repo}/git/trees/main?recursive=1"

# NusaX names lexicon files in full English; NusaWrites uses ISO-ish codes.
LEX_NAME = {"acehnese": "ace", "balinese": "ban", "buginese": "bug",
            "minangkabau": "min", "javanese": "jav", "sundanese": "sun",
            "banjarese": "bjn", "madurese": "mad", "ngaju": "nij",
            "toba_batak": "bbc", "indonesian": "ind", "english": "eng"}
WANT = {"ace", "ban", "bug", "min", "mak", "jav", "ind"}


def tree(repo: str) -> list[str]:
    r = requests.get(TREE.format(repo=repo), timeout=90)
    r.raise_for_status()
    return [x["path"] for x in r.json().get("tree", []) if x["type"] == "blob"]


def fetch_csv(repo: str, path: str) -> list[dict]:
    r = requests.get(RAW.format(repo=repo, path=path), timeout=120)
    if r.status_code != 200:
        return []
    return list(csv.DictReader(io.StringIO(r.text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/raw/nusa_github")
    a = ap.parse_args()
    out = ROOT / a.out
    results = []

    # ---- NusaX bilingual lexicons -------------------------------------
    paths = tree("IndoNLP/nusax")
    lex = [p for p in paths if "/lexicon/" in p and p.endswith(".csv")]
    print(f"NusaX lexicons: {len(lex)} files")
    for p in lex:
        name = Path(p).stem.lower()
        code = LEX_NAME.get(name)
        if code not in WANT:
            continue
        rows = fetch_csv("IndoNLP/nusax", p)
        if not rows:
            print(f"  {name}: FETCH FAILED")
            continue
        d = out / "lexicon"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{code}.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {code:4s} ({name:14s}) {len(rows):>6,} entries  "
              f"cols={list(rows[0])[:4]}")
        results.append({"kind": "lexicon", "lang": code, "rows": len(rows),
                        "path": str(f.relative_to(ROOT)), "src": p})

    # ---- NusaWrites parallel + paragraphs ------------------------------
    paths = tree("IndoNLP/nusa-writes")
    for family, kind in (("nusa_kalimat-mt", "parallel"),
                         ("nusa_alinea-paragraph", "paragraph")):
        sel = [p for p in paths
               if p.startswith("data/" + family) and p.endswith(".csv")]
        print(f"\nNusaWrites {family}: {len(sel)} files")
        by_lang: dict[str, list[dict]] = {}
        for p in sel:
            stem = Path(p).stem                    # family-<lang>-<split>
            parts = stem.split("-")
            lang, split = parts[-2], parts[-1]
            if lang not in WANT:
                continue
            rows = fetch_csv("IndoNLP/nusa-writes", p)
            for r in rows:
                r["_split"] = split
            by_lang.setdefault(lang, []).extend(rows)
        for lang, rows in sorted(by_lang.items()):
            if not rows:
                continue
            d = out / kind
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{lang}.jsonl"
            with open(f, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            chars = sum(len(str(v)) for r in rows for v in r.values())
            print(f"  {lang:4s} {len(rows):>7,} rows  {chars/1e6:6.2f} MB  "
                  f"cols={[c for c in rows[0] if not c.startswith('_')][:5]}")
            results.append({"kind": kind, "lang": lang, "rows": len(rows),
                            "chars": chars, "path": str(f.relative_to(ROOT))})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "exp": "S2.5_nusa_github",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Pulled from GitHub because the indonlp/ and SEACrowd/ Hub "
                 "repos are script-based and dead under datasets 4.x (F32)."),
        "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
