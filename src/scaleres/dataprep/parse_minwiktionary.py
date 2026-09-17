#!/usr/bin/env python
"""Parse the Minangkabau Wiktionary dump into dataset/raw/dict/min/minwiktionary*.jsonl.

The dump is template-regular, so this is a parser, not a model call:

  =={{bahasa|min}}==  ... {{-n-|min}} / {{-v-|min}} ...   one Minangkabau entry
      # definition                  monolingual Minangkabau definition, or [[link]]
      #* ''example'' / #:{{id}} ''translation''
      {{-syn-}} [[a]],[[b]]         synonyms
      * {{bhs|id}}: {{t|id|perut}}  translations (id / en)
  =={{bahasa|id}}==  ... # [[mamaluak]]   Indonesian headword -> Minangkabau word(s)

Outputs (schema of the other dict files, extra carries the rest):
  minwiktionary.jsonl          min headword -> ind translation   (from {{t|id|..}} or id entries)
  minwiktionary_defs.jsonl     min headword -> definition text    (translation_lang "min_or_ind")

    python -m scaleres.dataprep.parse_minwiktionary
"""
from __future__ import annotations

import bz2
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DUMP = ROOT / "dataset/raw/dict/_source/minwiktionary-latest-pages-articles.xml.bz2"
OUT = ROOT / "dataset/raw/dict/min"

LANG_HDR = re.compile(r"^==\s*\{\{bahasa\|([a-z-]+)\}\}\s*==\s*$", re.M)
POS = re.compile(r"^\{\{-([a-z]+)-\|(?:min|id)\}\}", re.M)
TRANS = re.compile(r"\{\{t\|(id|en)\|([^}|]+)")
LINK = re.compile(r"\[\[(?:[^]|]*\|)?([^]]+)\]\]")
LABEL = re.compile(r"\{\{lb\|min\|([^}]+)\}\}")


def clean(s: str) -> str:
    s = LINK.sub(r"\1", s)
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip(" ;,.")


def sections(text: str):
    heads = list(LANG_HDR.finditer(text))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        yield m.group(1), text[m.end():end]


def main():
    ind_pairs, defs = [], []
    stats = Counter()
    for _, el in ET.iterparse(bz2.open(DUMP), events=("end",)):
        if not el.tag.endswith("}page"):
            continue
        ns, title, text = (el.find(".//{*}" + t) for t in ("ns", "title", "text"))
        if ns is None or ns.text != "0" or text is None or not text.text:
            el.clear(); continue
        title, body = title.text.strip(), text.text
        for lang, sec in sections(body):
            stats[f"section:{lang}"] += 1
            pos = (POS.search(sec).group(1) if POS.search(sec) else None)
            labels = LABEL.findall(sec)
            if lang == "min":
                for line in sec.splitlines():
                    if re.match(r"^#(?![*:])", line):
                        d = clean(line[1:])
                        if d:
                            # min.wiktionary definitions are written in Minangkabau OR
                            # Indonesian, often both; not separable reliably on short text
                            defs.append({"headword": title, "translation": d, "translation_lang": "min_or_ind",
                                         "pos": pos, "source": "minwiktionary",
                                         "extra": {"dialect_labels": labels} if labels else {}})
                syn = []
                m = re.search(r"\{\{-syn-\}\}(.*?)(?=\n\{\{-|\Z)", sec, re.S)
                if m:
                    syn = [clean(x) for x in LINK.findall(m.group(1))]
                for tl, w in TRANS.findall(sec):
                    if tl == "id":
                        ind_pairs.append({"headword": title, "translation": clean(w), "translation_lang": "ind",
                                          "pos": pos, "source": "minwiktionary",
                                          "extra": {"via": "t|id", **({"synonyms": syn} if syn else {}),
                                                    **({"dialect_labels": labels} if labels else {})}})
            elif lang == "id":
                # Indonesian headword whose definition lines are Minangkabau equivalents
                for line in sec.splitlines():
                    if re.match(r"^#(?![*:])", line):
                        # only a line that is NOTHING BUT links is an equivalents list;
                        # a link inside a descriptive definition is not a translation
                        body_ = re.sub(r"\{\{[^}]*\}\}", "", line[1:])
                        if LINK.sub("", body_).strip(" ,;/.") != "":
                            stats["id_line_descriptive_skipped"] += 1
                            continue
                        for w in (v for l in LINK.findall(body_) for v in re.split(r"[;,/]", l)):
                            w = clean(w)
                            if w and len(w.split()) <= 4:
                                ind_pairs.append({"headword": w, "translation": title, "translation_lang": "ind",
                                                  "pos": pos, "source": "minwiktionary",
                                                  "extra": {"via": "id-entry"}})
        el.clear()

    def dump(rows, name):
        seen, out = set(), []
        for r in rows:
            k = (r["headword"].lower(), r["translation"].lower())
            if k not in seen and r["headword"] and r["translation"]:
                seen.add(k); out.append(r)
        with open(OUT / name, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return out

    a = dump(ind_pairs, "minwiktionary.jsonl")
    b = dump(defs, "minwiktionary_defs.jsonl")
    via = Counter(r["extra"]["via"] for r in a)
    print({k: v for k, v in stats.most_common(6)})
    print(f"min->ind pairs {len(a):,} ({dict(via)}), headwords {len({r['headword'].lower() for r in a}):,}")
    print(f"min definitions {len(b):,}, headwords {len({r['headword'].lower() for r in b}):,}")


if __name__ == "__main__":
    main()
