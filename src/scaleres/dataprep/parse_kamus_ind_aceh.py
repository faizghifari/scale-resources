#!/usr/bin/env python
"""S2.16 -- rule-parse the Kamus Indonesia-Aceh OCR into entries, without the VLM.

Unlike the Aceh->Indonesia volumes (running prose with example sentences, which needs the
model), this volume is one entry per paragraph in a fixed shape:

    <indonesian headword> <pos> <acehnese equivalent>; <equivalent>; ...
    kepada p keu; ba'
    merbah n brujue'; beurujue'; --- besar brujue' balee; --- kecil brujue' breueh

so a regex reads it exactly and in seconds, where the VLM needs ~10 hours of server time
for the 1176 pages and adds its own transcription risk. `---` stands for the headword.

Writes vlm_parse-compatible JSON (same keys flatten_aceh_dict expects) to
  dataset/raw/dict/_source/archive_org/rule_parse/ind_aceh/page_NNNN.json

    python -m scaleres.dataprep.parse_kamus_ind_aceh
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "dataset/raw/dict/_source/archive_org"
SRC = BASE / "vlm_ocr/ind_aceh"
OUT = BASE / "rule_parse/ind_aceh"
# Only real page files: the repair pass leaves page_NNNN.prerepair.txt backups beside them,
# and a bare page_* glob reads those as extra pages of superseded text.
PAGE_FILE = re.compile(r"page_\d+\.(txt|json)")
RESULTS = ROOT / "autoresearch/experiments/results"

# part-of-speech abbreviations used by the dictionary, longest first so "adv" wins over "a"
POS = ["adv", "num", "pron", "conj", "part", "pref", "suf", "n", "v", "a", "p"]
POS_RE = re.compile(rf"^(?P<hw>.{{1,40}}?)\s+(?P<pos>{'|'.join(POS)})\s+(?P<rest>\S.*)$")
SENSE = re.compile(r"^\d+\.\s*")
HEADER = re.compile(r"^(INDONESIA\s+ACEH|ACEH\s+INDONESIA|\d+|[A-Z])$")
DASH = re.compile(r"-{2,3}")
# Indonesian affix markers that make a sub-entry a derived form rather than a compound
AFFIX = re.compile(r"^(ber|mem|men|meng|meny|me|ter|di|per|pe|se|ke)\b|\b(kan|an|i|nya)$")
PREFIX = ["memper", "menyer", "meng", "meny", "mem", "men", "ber", "ter", "per", "di", "me", "se", "ke", "pe"]
SUFFIX = ["kannya", "annya", "kan", "an", "nya", "i"]
AFFIX_PRE = {"ber", "mem", "men", "meng", "meny", "me", "ter", "di", "per", "pe", "se", "ke",
             "memper", "diper", "berke", "keter"}
AFFIX_SUF = {"an", "kan", "i", "nya", "kannya", "annya"}


def split_trailing_affix(chunk: str):
    """("alamat; tanda; ber", ...) -> ("alamat; tanda", "ber")."""
    toks = chunk.split()
    if toks and toks[-1].lower() in AFFIX_PRE:
        return " ".join(toks[:-1]).rstrip(" ;,"), toks[-1].lower()
    return chunk, None


def join_affix(a: str, b: str) -> str:
    """Glue an affix to the headword: single-word headwords take it directly."""
    return a + b if " " not in a.strip() else f"{a.strip()} {b}"


def derived_of(item: str, hw: str):
    """Is `item` an Indonesian inflected form of the headword written inline?

    The volume prints `kembang v berkembang keumang; ...`, so the first words of an
    equivalent list can be an Indonesian derived form whose own Acehnese glosses follow.
    Returns (form, remainder) or None. Spaces are ignored because the OCR sometimes
    breaks a word ("menggan yang" for "mengganyang").
    """
    flat = lambda s: re.sub(r"[^a-z']", "", s.lower())
    target = flat(hw)
    toks = item.split()
    for n in (1, 2, 3):
        if len(toks) <= n:
            break
        cand = flat(" ".join(toks[:n]))
        if not cand or cand == target:
            continue
        for p in [""] + PREFIX:
            if p and not cand.startswith(p):
                continue
            stem = cand[len(p):]
            for s in [""] + SUFFIX:
                if s and not stem.endswith(s):
                    continue
                if (stem[:-len(s)] if s else stem) == target and (p or s):
                    return " ".join(toks[:n]), " ".join(toks[n:])
    return None


def blocks(text: str):
    for b in re.split(r"\n\s*\n", text):
        b = " ".join(b.split())
        if not b or HEADER.match(b):
            continue
        yield b


def clean_ace(s: str) -> list[str]:
    """Split the Acehnese side into individual equivalents."""
    out = []
    for part in s.split(";"):
        part = SENSE.sub("", part.strip())
        part = re.sub(r"\((?:[^)]*)\)", " ", part)          # parenthetical variants/notes
        part = re.sub(r"\b(ki|cak|kas|hik|kac|mele|fare)\b", " ", part)   # usage labels
        part = " ".join(part.split()).strip(" ,;:.-")
        if 1 <= len(part) <= 60 and not part.isdigit():
            out.append(part)
    return out


def parse_block(b: str, stats: collections.Counter):
    """-> (entry dict or None). Sub-entries after a `---` marker become `derived`."""
    b = SENSE.sub("", b)
    m = POS_RE.match(b)
    if not m:
        stats["no_pos_marker"] += 1
        return None
    hw = " ".join(m.group("hw").split()).strip(" ,;:.")
    if not hw or DASH.search(hw) or len(hw.split()) > 3:
        stats["bad_headword"] += 1
        return None
    head, *subs = DASH.split(m.group("rest"))
    main, derived = [], []
    # "alamat n alamat; tanda; ber --- 1. mupat" -- the affix sits BEFORE the dash, so the
    # fragment trailing one chunk belongs to the sub-entry that follows it ("beralamat").
    head, pend = split_trailing_affix(head)
    # an inline Indonesian derived form claims every equivalent up to the next such form
    current = None
    for item in head.split(";"):
        item = SENSE.sub("", " ".join(item.split()))
        if not item:
            continue
        d = derived_of(item, hw)
        if d:
            current, item = d[0], SENSE.sub("", d[1].strip())
            stats["inline_derived"] += 1
        for a in clean_ace(item):
            (derived.append({"form_ind": current, "ace": a}) if current else main.append(a))
    for s in subs:
        s, nxt = split_trailing_affix(" ".join(s.split()))
        s = SENSE.sub("", s.strip())
        first, _, tail = s.partition(" ")
        if pend:                                 # ber --- -> beralamat, whole chunk is the gloss
            form, content = join_affix(pend, hw), s
            if first.lower() in AFFIX_SUF:       # mem --- kan -> memalamatkan
                form, content = form + first.lower(), tail
        elif first.lower() in AFFIX_SUF:         # --- an -> alamatan
            form, content = join_affix(hw, first.lower()), tail
        else:                                    # --- baju -> alas baju (a compound entry)
            form, content = f"{hw} {first}", tail
        form = re.sub(r"\s*\d+\s*[.:]?\s*", " ", form).strip(" ,;:.-") or hw
        pend = nxt
        for a in clean_ace(content):
            derived.append({"form_ind": form, "ace": a})
        stats["sub_entries"] += 1
    # a derived form printed with no gloss after it ("anut v menganuti;") must not be kept
    # as if it were the Acehnese equivalent
    keep = [a for a in main if not (a.lower() in AFFIX_PRE or derived_of(a + " x", hw))]
    stats["dropped_indonesian_form"] += len(main) - len(keep)
    main = keep
    if not main and not derived:
        stats["no_equivalents"] += 1
        return None
    stats["entries"] += 1
    stats["equivalents"] += len(main)
    stats["derived"] += len(derived)
    return {"headword_ind": hw, "pos": m.group("pos"), "ace": main,
            "derived_ind": derived, "examples": [], "note": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N pages (for checking)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    files = sorted(f for f in SRC.glob("page_*.txt") if PAGE_FILE.fullmatch(f.name))
    if a.limit:
        files = files[:a.limit]
    for f in files:
        entries = [e for e in (parse_block(b, stats) for b in blocks(f.read_text(errors="ignore"))) if e]
        stats["pages"] += 1
        (OUT / f"{f.stem}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    # A garbled sub-entry line can leave the NEXT sub-headword sitting where its Acehnese
    # gloss should be ("--- persekutuan nelayan - -- asing ranto" on p203, where the gloss of
    # "daerah persekutuan" is missing from the scan). Those are recognisable: a one-word
    # Acehnese side that is itself an Indonesian headword of this dictionary and never appears
    # as an Acehnese equivalent anywhere. 1.0% of sub-entries; flagged, not dropped, because
    # ace/ind share real vocabulary.
    hw, ace_seen = set(), set()
    per_page = {}
    for f in sorted(f for f in OUT.glob("page_*.json") if PAGE_FILE.fullmatch(f.name)):
        per_page[f] = json.loads(f.read_text())
        for e in per_page[f]:
            hw.add(e["headword_ind"].lower())
            ace_seen.update(a.lower() for a in e["ace"])
    for f, entries in per_page.items():
        touched = False
        for e in entries:
            for d in e["derived_ind"]:
                a = d["ace"].lower()
                if " " not in a and a in hw and a not in ace_seen:
                    d["suspect_split"] = True
                    stats["derived_flagged_suspect"] += 1
                    touched = True
        if touched:
            f.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    rep = {"exp": "S2.16_rule_parse_ind_aceh",
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "counts": dict(stats), "unique_indonesian_headwords": len(hw)}
    (RESULTS / "S2_16_rule_parse_ind_aceh.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
