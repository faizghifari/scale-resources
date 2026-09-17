#!/usr/bin/env python
"""S2.15 -- flatten the parsed Aceh dictionary pages into dictionary entries + sentence pairs.

Input: dataset/raw/dict/_source/archive_org/vlm_parse/<vol>/page_*.json (see ocr_dictionary.py)
  jilid1 + jilid2 = Kamus Bahasa Aceh-Indonesia (Acehnese headword -> Indonesian gloss)
  ind_aceh       = Kamus Indonesia-Aceh          (Indonesian headword -> Acehnese equivalents)

Outputs, in the schema the other dictionary files use (headword is ALWAYS the Acehnese side):
  dataset/raw/dict/ace/kamus_aceh_indonesia.jsonl
  dataset/raw/dict/ace/kamus_indonesia_aceh.jsonl
  dataset/raw/parallel_src/kamus_aceh_examples/ace.jsonl   example sentences, ace <-> ind
      (picked up by build_parallel_combined, which screens and decontaminates them)

Two OCR artefacts are repaired here rather than in the prompt, because both need corpus
evidence the model does not have:
  * line-break hyphens: "lu-bangnya" -> "lubangnya" only when the joined form is attested
    elsewhere in the volume more often than the hyphenated one (keeps daun-daunan intact).
  * `continuation` records (an entry split across a page break) are appended to the last
    entry of the previous page.

    python -m scaleres.dataprep.flatten_aceh_dict
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
DICT = ROOT / "dataset/raw/dict/ace"
PAR = ROOT / "dataset/raw/parallel_src/kamus_aceh_examples"
RESULTS = ROOT / "autoresearch/experiments/results"
ACE_IND = ["jilid1", "jilid2"]
WORD = re.compile(r"[A-Za-zÀ-ÿ'’]+")
# Only real page files. The repair pass keeps backups beside them as page_NNNN.prerepair.txt,
# which a bare page_* glob swallows: the rule parser read 1,180 "pages" of a 1,176-page
# volume and wrote page_0012.prerepair.json, feeding superseded pre-repair text back in.
PAGE_FILE = re.compile(r"page_\d+\.(txt|json)")
# in both volumes a dash stands in for the headword inside examples and sub-senses
DASH = re.compile(r"(?<![A-Za-zÀ-ÿ])-{1,3}(?![A-Za-zÀ-ÿ0-9-])")


def expand_dash(text: str, headword: str) -> str:
    """"ayah geu - si Gam u peukan" + "hue" -> "ayah geu hue si Gam u peukan"."""
    hw = headword.strip()
    return DASH.sub(hw, text) if hw and not hw.startswith("-") else text


def expand_affix_dash(form: str, headword: str) -> str:
    """A derived form prints the affix with an attached dash: "meu-" -> "meupheueng"."""
    hw = headword.split(",")[0].strip()
    if not hw:
        return form
    f = form.strip()
    if re.fullmatch(r"[A-Za-zÀ-ÿ']{1,5}-", f):
        return f[:-1] + hw
    if re.fullmatch(r"-[A-Za-zÀ-ÿ']{1,5}", f):
        return hw + f[1:]
    return form


def example_is_usable(ace: str, ind: str) -> bool:
    """Reject a gloss dumped into one field, but keep short phrase pairs.

    Most examples in these volumes are phrases, not sentences: the entry for `rugoe`
    gives "lôn ka -" / "Saya sudah rugi", which is a real pair of three words a side once
    the placeholder is expanded. An earlier >=4-words-a-side rule read those as
    column-bleed and dropped two thirds of the examples. What actually marks bleed is an
    unresolved placeholder left on the Indonesian side ("bungong -, putik kembang"),
    since the dash there stands for a word from the Acehnese column.
    build_parallel_combined screens the survivors again with GlotLID and length ratio.
    """
    aw, iw = len(ace.split()), len(ind.split())
    if aw < 2 or iw < 2 or aw > 40 or iw > 40:
        return False
    if DASH.search(ind):
        return False
    if not 0.25 <= aw / iw <= 3.0:      # Indonesian glosses run wordier than Acehnese
        return False
    return ace.count(",") <= 3 and ind.count(",") <= 5


def example_unit(ace: str, ind: str) -> str:
    return "sentence" if min(len(ace.split()), len(ind.split())) >= 4 else "phrase"


def volume_word_counts(vols) -> collections.Counter:
    c = collections.Counter()
    for v in vols:
        for f in sorted(f for f in (BASE / "vlm_ocr" / v).glob("page_*.txt")
                        if PAGE_FILE.fullmatch(f.name)):
            c.update(w.lower() for w in WORD.findall(f.read_text()))
    return c


def dehyphenate(text: str, counts: collections.Counter) -> str:
    def fix(m):
        a, b = m.group(1), m.group(2)
        joined, hyph = (a + b).lower(), f"{a.lower()}-{b.lower()}"
        return (a + b) if counts[joined] > counts[hyph] else m.group(0)
    text = re.sub(r"\b([A-Za-zÀ-ÿ']{2,})-([A-Za-zÀ-ÿ']{2,})\b", fix, text)

    # Inside an example the OCR keeps the printed end-of-line hyphen but folds the line,
    # so the two halves arrive space-separated ("Ham- zah", "ban- dum", "tameu- tang").
    # A mid-phrase hyphen is not otherwise written here, so joining is right whenever the
    # result is a word this volume actually uses; short prefixes that join into nothing
    # are left alone, since those are the "meu-" placeholder convention.
    def fix_spaced(m):
        a, b = m.group(1), m.group(2)
        return (a + b) if counts[(a + b).lower()] >= 2 else m.group(0)
    return re.sub(r"\b([A-Za-zÀ-ÿ']{2,})-\s+([A-Za-zÀ-ÿ']{2,})\b", fix_spaced, text)


def parse_dir(vol) -> Path:
    """ind_aceh is read deterministically (parse_kamus_ind_aceh); the others need the VLM."""
    return BASE / ("rule_parse" if vol == "ind_aceh" else "vlm_parse") / vol


def unreliable_pages(vol) -> set:
    """Pages the OCR QC still flags after the repair pass, as page numbers.

    A handful of scans (jilid1 p211, p381) are mirrored show-through from the reverse of
    the sheet: no orientation or contrast variant makes them legible, and the model returns
    plausible-looking nonsense ("ub -> but yasto, agisel neb"). Word-level coverage cannot
    see this, because it scores whether each word exists in the volume, not whether the
    line means anything -- so these pages must be excluded by name rather than by score.
    """
    f = RESULTS / "S2_14_ocr_qc.json"
    if not f.exists():
        return set()
    return set(json.loads(f.read_text()).get(vol, {}).get("suspects", []))


def pages(vol, stats=None):
    skip = unreliable_pages(vol)
    for f in sorted(f for f in parse_dir(vol).glob("page_*.json")
                    if PAGE_FILE.fullmatch(f.name)):
        if int(f.stem.split("_")[1]) in skip:
            if stats is not None:
                stats[f"pages_skipped_unreliable_{vol}"] += 1
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            yield f, None; continue
        yield f, data if isinstance(data, list) else [data]


def flatten_ace_ind(counts, stats):
    """Acehnese -> Indonesian volumes. Yields (dict rows, example pairs)."""
    rows, pairs = [], []
    last = None
    for vol in ACE_IND:
        for f, data in pages(vol, stats):
            if data is None:
                stats["bad_json_files"] += 1; continue
            for e in data:
                if not isinstance(e, dict):
                    stats["non_dict_item"] += 1; continue
                if e.get("non_entry_page"):
                    stats["non_entry_page"] += 1; continue
                if e.get("continuation"):
                    stats["continuation"] += 1
                    if last is not None:
                        extra = dehyphenate(str(e["continuation"]), counts).strip()
                        last["translation"] = (last["translation"] + " " + extra).strip()
                    continue
                hw = dehyphenate(str(e.get("headword") or "").strip(), counts)
                if not hw:
                    stats["no_headword"] += 1; continue
                gloss = expand_dash(dehyphenate(str(e.get("gloss_ind") or "").strip(), counts), hw)
                see = [s for s in (e.get("see_also") or []) if isinstance(s, str)]
                if gloss:
                    last = {"headword": hw, "translation": gloss, "translation_lang": "ind",
                            "pos": e.get("pos"), "source": "kamus_aceh_indonesia",
                            "extra": {"volume": vol, "page": f.stem, "homograph": e.get("homograph"),
                                      "see_also": see or None}}
                    rows.append(last); stats["entries"] += 1
                elif see:
                    rows.append({"headword": hw, "translation": "", "translation_lang": "ind",
                                 "pos": None, "source": "kamus_aceh_indonesia",
                                 "extra": {"volume": vol, "page": f.stem, "cross_reference_only": True,
                                           "see_also": see}})
                    stats["cross_ref_only"] += 1
                for d in (e.get("derived") or []):
                    if isinstance(d, dict) and d.get("form") and d.get("gloss_ind"):
                        # the dash in a derived form stands for the headword too: "nu -" -> "nu majun"
                        rows.append({"headword": expand_affix_dash(
                                         expand_dash(dehyphenate(str(d["form"]).strip(), counts), hw), hw),
                                     "translation": expand_dash(dehyphenate(str(d["gloss_ind"]).strip(), counts), hw),
                                     "translation_lang": "ind", "pos": None,
                                     "source": "kamus_aceh_indonesia",
                                     "extra": {"volume": vol, "page": f.stem, "derived_from": hw}})
                        stats["derived"] += 1
                for ex in (e.get("examples") or []):
                    if isinstance(ex, dict) and ex.get("ace") and ex.get("ind"):
                        a = expand_dash(dehyphenate(str(ex["ace"]).strip(), counts), hw)
                        # the Indonesian side is NOT expanded with hw: its dash stands for a
                        # word from the Acehnese column, so filling in the Acehnese headword
                        # would plant Acehnese in the Indonesian field
                        i = dehyphenate(str(ex["ind"]).strip(), counts)
                        stats["examples_seen"] += 1
                        if not example_is_usable(a, i):
                            stats["examples_dropped_not_sentence"] += 1; continue
                        pairs.append({"ace": a, "ind": i, "headword": hw,
                                      "unit": example_unit(a, i),
                                      "volume": vol, "page": f.stem})
                        stats["examples"] += 1
    return rows, pairs


def flatten_ind_ace(counts, stats):
    rows, pairs = [], []
    for f, data in pages("ind_aceh", stats):
        if data is None:
            stats["bad_json_files"] += 1; continue
        for e in data:
            if not isinstance(e, dict) or e.get("non_entry_page") or e.get("continuation"):
                stats["skipped_ind_aceh"] += 1; continue
            ind = dehyphenate(str(e.get("headword_ind") or "").strip(), counts)
            aces = e.get("ace") or []
            aces = [aces] if isinstance(aces, str) else aces
            for a in aces:
                a = dehyphenate(str(a).strip(), counts)
                if ind and a:
                    rows.append({"headword": a, "translation": ind, "translation_lang": "ind",
                                 "pos": e.get("pos"), "source": "kamus_indonesia_aceh",
                                 "extra": {"page": f.stem, "note": e.get("note")}})
                    stats["entries_ind_aceh"] += 1
            # sub-entries and inflected forms: "beralamat" -> "mupat", "alas baju" -> "leupeh bajee"
            for d in (e.get("derived_ind") or []):
                if not isinstance(d, dict):
                    continue
                form = dehyphenate(str(d.get("form_ind") or "").strip(), counts)
                a = dehyphenate(str(d.get("ace") or "").strip(), counts)
                if form and a:
                    rows.append({"headword": a, "translation": form, "translation_lang": "ind",
                                 "pos": None, "source": "kamus_indonesia_aceh",
                                 "extra": {"page": f.stem, "derived_from": ind,
                                           "suspect_split": d.get("suspect_split")}})
                    stats["derived_ind_aceh"] += 1
                    stats["derived_suspect"] += bool(d.get("suspect_split"))
            for ex in (e.get("examples") or []):
                if isinstance(ex, dict) and ex.get("ace") and ex.get("ind"):
                    a = dehyphenate(str(ex["ace"]).strip(), counts)
                    i = expand_dash(dehyphenate(str(ex["ind"]).strip(), counts), ind)
                    stats["examples_ind_aceh_seen"] += 1
                    if not example_is_usable(a, i):
                        stats["examples_ind_aceh_dropped"] += 1; continue
                    pairs.append({"ace": a, "ind": i, "headword": ind,
                                  "unit": example_unit(a, i),
                                  "volume": "ind_aceh", "page": f.stem})
                    stats["examples_ind_aceh"] += 1
    return rows, pairs


# Indonesian closed-class words. Acehnese borrows Indonesian content words freely, so an
# equivalent identical to an Indonesian noun is usually a real loanword, but the function
# words below are never the Acehnese gloss of an entry: where one shows up in the Acehnese
# column it came from an Indonesian explanatory phrase inside the equivalent list
# ("aman; tidak ada gangguan" -> "tidak" harvested as an equivalent).
IND_FUNCTION = set("""yang dan dengan untuk tidak adalah akan dari pada dalam atau kepada
oleh sudah sedang juga bisa dapat harus telah belum masih lebih paling seperti karena
sehingga tetapi namun saya kamu dia mereka kami kita ini itu ada bagi serta agar supaya
jika kalau ketika setelah sebelum sambil hanya saja pun bahwa ialah yaitu adapun maupun
melainkan walaupun meskipun sebab hingga sampai antara terhadap tentang menurut
sangat amat sekali begitu demikian tersebut setiap semua seluruh beberapa banyak""".split())


def valid_headword(hw: str) -> bool:
    """Reject dash artefacts like "nu -" and Indonesian function words in the Aceh column."""
    hw = hw.strip()
    if hw.lower() in IND_FUNCTION:
        return False
    return len(hw) >= 2 and len(re.findall(r"[A-Za-zÀ-ÿ]", hw)) >= 2 and "-" not in hw.split()


def write_jsonl(path, rows, stats=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if "headword" in r and not valid_headword(r["headword"]):
                if stats is not None:
                    stats["dropped_bad_headword"] += 1
                continue
            k = json.dumps([r.get("headword"), r.get("translation"), r.get("ace"), r.get("ind")], ensure_ascii=False).lower()
            if k in seen:
                continue
            seen.add(k)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pages", type=int, default=100,
                    help="refuse to flatten if a volume has fewer parsed pages than this")
    a = ap.parse_args()
    stats = collections.Counter()
    have = {v: sum(1 for f in parse_dir(v).glob("page_*.json") if PAGE_FILE.fullmatch(f.name))
            for v in ["jilid1", "jilid2", "ind_aceh"]}
    print("parsed pages:", have)
    if sum(have.values()) < a.min_pages:
        raise SystemExit(f"only {sum(have.values())} parsed pages -- run the parse stage first")
    counts = volume_word_counts(["jilid1", "jilid2", "ind_aceh"])
    rows_a, pairs_a = flatten_ace_ind(counts, stats)
    rows_b, pairs_b = flatten_ind_ace(counts, stats)
    n_a = write_jsonl(DICT / "kamus_aceh_indonesia.jsonl", rows_a, stats)
    n_b = write_jsonl(DICT / "kamus_indonesia_aceh.jsonl", rows_b, stats)
    n_p = write_jsonl(PAR / "ace.jsonl", pairs_a + pairs_b, stats)
    hw = {r["headword"].lower() for r in rows_a + rows_b if valid_headword(r["headword"])}
    rep = {"exp": "S2.15_flatten_aceh_dict", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "parsed_pages": have, "counts": dict(stats),
           "written": {"kamus_aceh_indonesia.jsonl": n_a, "kamus_indonesia_aceh.jsonl": n_b,
                       "kamus_aceh_examples/ace.jsonl": n_p},
           "unique_acehnese_headwords": len(hw)}
    (RESULTS / "S2_15_flatten_aceh_dict.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
