#!/usr/bin/env python
"""S2.17 -- combine every per-language dictionary source into one deduplicated file.

Reads dataset/raw/dict/<lang>/*.jsonl (all sources share one schema: headword, translation,
translation_lang, pos, source, extra) and writes

  dataset/dict/<lang>/combined_v1.jsonl        word/phrase equivalents, one row per pair
  dataset/dict/<lang>/definitions_v1.jsonl     rows whose "translation" is a definition,
                                               not an equivalent (kaikki, minwiktionary_defs)

Equivalents and definitions are split because they are used differently: the pair list feeds
lexical scoring and prompt construction, while definitions are prose and would pollute it.

Sources are merged in SOURCE_ORDER so that when the same pair appears twice the row kept is
the one from the more trustworthy source, and `also_in` records the rest.

    python -m scaleres.dataprep.build_dict_combined
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "dataset/raw/dict"
OUT = ROOT / "dataset/dict"
RESULTS = ROOT / "autoresearch/experiments/results"
LANGS = ["min", "ace", "bug", "mak"]

# hand-built lexicons first, crowd/scraped next, automatically induced last
SOURCE_ORDER = ["kamus_indonesia_aceh", "kamus_aceh_indonesia", "minangNLP", "minwiktionary",
                "NusaX", "abvd", "panlex", "kaikki", "google/smol", "minwiktionary_defs"]
# a "translation" longer than this, or one with sentence punctuation, is a definition
DEF_CHARS = 60
DEF_SOURCES = {"kaikki", "minwiktionary_defs"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'’-]+", " ", s)).strip()


def is_definition(row) -> bool:
    t = str(row.get("translation") or "")
    # a gloss written in the language itself, or one whose language is undecided
    # ("min_or_ind" from parse_minwiktionary), is never a translation pair
    if row.get("translation_lang") not in ("ind", "eng"):
        return True
    if row.get("source") in DEF_SOURCES and (len(t) > DEF_CHARS or "," in t or ";" in t):
        return True
    return len(t) > DEF_CHARS


def rank(src: str) -> int:
    return SOURCE_ORDER.index(src) if src in SOURCE_ORDER else len(SOURCE_ORDER)


def combine(lang: str, stats: collections.Counter):
    rows = []
    for f in sorted((RAW / lang).glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    stats[f"{lang}:bad_json"] += 1
                    continue
                if not (r.get("headword") and r.get("translation")):
                    stats[f"{lang}:empty"] += 1
                    continue
                rows.append(r)
    rows.sort(key=lambda r: rank(r.get("source", "")))
    pairs, defs = {}, {}
    for r in rows:
        bucket = defs if is_definition(r) else pairs
        key = (norm(r["headword"]), norm(r["translation"]), r.get("translation_lang"))
        if not key[0] or not key[1]:
            stats[f"{lang}:empty_after_norm"] += 1
            continue
        # loanwords spelled the same in both languages are real but carry no lexical signal
        if key[0] == key[1]:
            r = dict(r, identity=True)
            stats[f"{lang}:identity_pair"] += 1
        if key in bucket:
            also = bucket[key].setdefault("also_in", [])
            if r.get("source") not in also and r.get("source") != bucket[key].get("source"):
                also.append(r.get("source"))
            stats[f"{lang}:duplicate"] += 1
            continue
        bucket[key] = dict(r)
    return list(pairs.values()), list(defs.values())


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS)
    a = ap.parse_args()
    stats = collections.Counter()
    rep = {"exp": "S2.17_dict_combined",
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "langs": {}}
    for lang in a.langs:
        pairs, defs = combine(lang, stats)
        n_p = write(OUT / lang / "combined_v1.jsonl", pairs)
        n_d = write(OUT / lang / "definitions_v1.jsonl", defs)
        by_src = collections.Counter(r.get("source") for r in pairs)
        tgt = collections.Counter(r.get("translation_lang") for r in pairs)
        rep["langs"][lang] = {"pairs": n_p, "definitions": n_d,
                              "identity_pairs": sum(1 for r in pairs if r.get("identity")),
                              "unique_headwords": len({norm(r["headword"]) for r in pairs}),
                              "by_source": dict(by_src.most_common()),
                              "by_translation_lang": dict(tgt.most_common())}
        print(f"{lang}: {n_p:7d} pairs  {n_d:6d} definitions  "
              f"{rep['langs'][lang]['unique_headwords']:6d} headwords  {dict(by_src.most_common(4))}")
    rep["counts"] = dict(stats)
    (RESULTS / "S2_17_dict_combined.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"wrote {RESULTS / 'S2_17_dict_combined.json'}")


if __name__ == "__main__":
    main()
