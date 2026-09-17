#!/usr/bin/env python
"""S2.11 -- one combined parallel dataset per expansion language.

Sources are streamed in TIER order (human > human_unverified > auto_aligned >
mined), so dedup on the normalised target keeps the best-provenance copy without
holding the 4M+ NLLB pairs in memory. Every tier goes through the same filters:

  a) length: non-empty, tgt >= 15 chars, len(tgt)/len(src) in [0.33, 3.0]
  b) untranslated: normalised src == normalised tgt
  c) dedup: exact (src, tgt), then one pair per normalised tgt
  d) mined only: LASER score >= 1.06
  e) target-side GlotLID margin screen, same config as screen_language_corpus
  f) eval decontamination: drop if tgt shares a 10-gram with dataset/eval/<lang>/*

Hard exclusion, enforced by never loading them: FLORES/FLORES+, sib200,
NusaX-MT, NusaX-senti -- they are the eval floor.

    python -m scaleres.dataprep.build_parallel_combined --langs bug min ace mak
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scaleres.dataprep.screen_language_corpus import LANGS, Screener, eval_ngrams, ngrams, norm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "dataset/raw/parallel_src"
OUT = ROOT / "dataset/parallel/combined"
RESULTS = ROOT / "autoresearch/experiments/results"
TIERS = ["human", "human_unverified", "auto_aligned", "mined"]
LASER_MIN = 1.06


def h(s: str) -> bytes:
    return hashlib.blake2b(s.encode(), digest_size=8).digest()


# ------------------------------------------------------------------ loaders
# each yields (src_lang, src_text, tgt_text, config, laser_score, extra)

def prosa(lang):
    for split in ["train", "validation", "test"]:
        with open(SRC / f"prosa_nusa_translation/{lang}/{split}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                yield "ind", r["original"], r["translated"], f"{lang}/{split}", None, {}


def nusawrites(lang):
    for l in open(ROOT / f"dataset/raw/nusa_github/parallel/{lang}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        yield "ind", r["ind_text"], r["tgt_text"], "nusa_kalimat-mt", None, {"id": r.get("id")}


def minangnlp(lang):
    d = SRC / "minangNLP/wiki_data"
    for split in ["train", "dev", "test"]:          # test_sent re-splits test; skipped
        s = open(d / f"src_{split}.txt", encoding="utf-8").read().split("\n")
        t = open(d / f"tgt_{split}.txt", encoding="utf-8").read().split("\n")
        assert len(s) == len(t), (split, len(s), len(t))
        for a, b in zip(s, t):                      # src_* is Minangkabau, tgt_* Indonesian
            yield "ind", b, a, split, None, {}


def idwiki(lang):
    with open(SRC / "indowikiparalelcorpora/indomin-parallel.csv", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)                                    # header wrongly says sundanese,indonesian
        for row in rd:
            if len(row) >= 2:
                yield "ind", row[1], row[0], "manualsets/indomin", None, {}


def hf_pairs(repo, cfg, src_col, tgt_col):
    def gen(lang):
        from datasets import load_dataset
        for r in load_dataset(repo, cfg, split="train"):
            yield "ind", r[src_col], r[tgt_col], cfg, None, {}
    return gen


def korpus_nusantara(sheets):
    def gen(lang):
        import pandas as pd
        for s in sheets:
            d = pd.read_excel(SRC / "korpus_nusantara/korpus nusantara.xlsx", sheet_name=s, header=None)
            for a, b in d.iloc[:, :2].itertuples(index=False):
                if isinstance(a, str) and isinstance(b, str):
                    yield "ind", a, b, s, None, {}
    return gen


def bhinneka(lang):
    import pandas as pd
    d = pd.read_excel(SRC / "bhinneka-korpus/parallel.xlsx")
    for r in d.itertuples(index=False):
        for sl in ["ind", "eng"]:
            if isinstance(getattr(r, sl), str) and isinstance(r.mak, str):
                yield sl, getattr(r, sl), r.mak, "parallel.xlsx", None, {"id": int(r.Id)}


def oldi(lang):
    from datasets import load_dataset
    eng = {r["id"]: r["text"] for r in load_dataset("openlanguagedata/oldi_seed", "eng_Latn", split="train")}
    for r in load_dataset("openlanguagedata/oldi_seed", f"{lang}_Latn", split="train"):
        if r["id"] in eng:
            yield "eng", eng[r["id"]], r["text"], f"{lang}_Latn", None, {"id": r["id"]}


def nllb(pair):
    def gen(lang):
        a, b = pair.split("-")
        tcol = 0 if a.startswith(lang) else 1
        sl = (b if tcol == 0 else a)[:3]
        with gzip.open(SRC / f"nllb/{pair}.gz", "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                try:
                    score = float(p[2])
                except ValueError:
                    continue
                yield sl, p[1 - tcol], p[tcol], pair, score, {"urls": [x for x in p[3:] if x.startswith("http")][:2]}
    return gen


def kamus_examples(lang):
    """Example sentences from the OCR'd Kamus Aceh-Indonesia (flatten_aceh_dict).

    Human-written by the dictionary's authors but read by a VLM, so the tier is
    human_unverified: the screener below still applies LID, length ratio and decontamination.
    """
    f = SRC / "kamus_aceh_examples" / f"{lang}.jsonl"
    if not f.exists():
        return
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            # `unit` distinguishes a full example sentence from a headword phrase
            # ("lôn ka rugoe" / "Saya sudah rugi"); both are real pairs, but a phrase
            # should not be counted as sentence-level parallel data downstream.
            yield "ind", r["ind"], r[lang], r.get("volume", "kamus"), None, {
                "headword": r.get("headword"), "page": r.get("page"),
                "unit": r.get("unit")}


SOURCES = {   # lang -> [(source name, tier, loader)]
    "bug": [("prosa-text/nusa-translation", "human", prosa),
            ("openlanguagedata/oldi_seed", "human", oldi),
            ("korpus-nusantara/bugis-wajo", "human_unverified", korpus_nusantara(["bugis wajo"])),
            ("korpus-nusantara/bugis-kelolao", "human_unverified", korpus_nusantara(["bugis kelolao"])),
            ("allenai/nllb", "mined", nllb("bug_Latn-eng_Latn"))],
    "min": [("prosa-text/nusa-translation", "human", prosa),
            ("IndoNLP/nusa-writes", "human", nusawrites),
            ("dindainastra/indowikiparalelcorpora", "human", idwiki),
            ("korpus-nusantara/padang", "human_unverified", korpus_nusantara(["padang"])),
            ("williamhtan/NusaBukuMT", "human_unverified", hf_pairs("williamhtan/NusaBukuMT", "id-min", "source_sentence", "target_sentence")),
            ("Exqrch/IndonesianNMT", "human_unverified", hf_pairs("Exqrch/IndonesianNMT", "id_min", "Indonesian", "Minangkabau")),
            ("fajri91/minangNLP", "auto_aligned", minangnlp),
            ("allenai/nllb", "mined", nllb("ind_Latn-min_Latn")),
            ("allenai/nllb", "mined", nllb("eng_Latn-min_Latn"))],
    "ace": [("openlanguagedata/oldi_seed", "human", oldi),
            ("kamus-aceh-indonesia/examples", "human_unverified", kamus_examples),
            ("allenai/nllb", "mined", nllb("ind_Latn-ace_Latn")),
            ("allenai/nllb", "mined", nllb("ace_Latn-eng_Latn"))],
    "mak": [("IndoNLP/nusa-writes", "human", nusawrites),
            ("joanitolopo/bhinneka-korpus", "human", bhinneka)],
}


def build(lang: str) -> dict:
    sc = Screener(LANGS[lang])
    ev = eval_ngrams(lang) if (ROOT / "dataset/eval" / lang).exists() else set()
    seen_pair, seen_tgt = set(), set()
    stats: dict[str, Counter] = defaultdict(Counter)
    out_dir = OUT / f"{lang}_v1"
    tmp = OUT / f"{lang}_v1.jsonl.partial"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    srcs = sorted(SOURCES[lang], key=lambda s: TIERS.index(s[1]))
    with open(tmp, "w", encoding="utf-8") as fo:
        for name, tier, loader in srcs:
            for sl, s, t, cfg, laser, extra in loader(lang):
                key = f"{name}|{cfg if tier == 'mined' else ''}"
                st = stats[key]
                st["raw"] += 1
                s, t = (s or "").strip(), (t or "").strip()
                if not s or len(t) < 15 or not (0.33 <= len(t) / max(len(s), 1) <= 3.0):
                    st["drop_length"] += 1; continue
                ns, nt = norm(s), norm(t)
                if ns == nt:
                    st["drop_untranslated"] += 1; continue
                if laser is not None and laser < LASER_MIN:
                    st["drop_laser"] += 1; continue
                hp, ht = h(ns + "\t" + nt), h(nt)
                if hp in seen_pair or ht in seen_tgt:
                    st["drop_dupe"] += 1; continue
                seen_pair.add(hp); seen_tgt.add(ht)
                ok, top, _ = sc.judge(t)
                if not ok:
                    st["drop_lid"] += 1; st[f"lid:{top}"] += 1; continue
                if ev and not ngrams(t).isdisjoint(ev):
                    st["drop_eval_overlap"] += 1; continue
                st["kept"] += 1; st["kept_tgt_chars"] += len(t)
                fo.write(json.dumps({"src_lang": sl, "src_text": s, "tgt_lang": lang, "tgt_text": t,
                                     "source": name, "config": cfg, "tier": tier,
                                     "laser_score": laser, "extra": json.dumps(extra, ensure_ascii=False)},
                                    ensure_ascii=False) + "\n")
            for k in sorted(k for k in stats if k.startswith(name)):
                v = stats[k]
                print(f"  {lang} {k:45s} {tier:16s} raw {v['raw']:>9,} kept {v['kept']:>8,}", flush=True)

    from datasets import Features, Value, load_dataset
    feats = Features({"src_lang": Value("string"), "src_text": Value("string"),
                      "tgt_lang": Value("string"), "tgt_text": Value("string"),
                      "source": Value("string"), "config": Value("string"),
                      "tier": Value("string"), "laser_score": Value("float64"),
                      "extra": Value("string")})
    ds = load_dataset("json", data_files=str(tmp), split="train", features=feats)
    ds.save_to_disk(str(out_dir))
    n = len(ds)
    tmp.unlink()

    by_tier, by_srclang = Counter(), Counter()
    for r in ds.select_columns(["tier", "src_lang"]):
        by_tier[r["tier"]] += 1; by_srclang[r["src_lang"]] += 1
    rep = {"exp": "S2.11_parallel_combined", "lang": lang,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "output": str(out_dir.relative_to(ROOT)), "pairs": n,
           "eval_ngrams": len(ev), "laser_min": LASER_MIN,
           "screen": {k: v for k, v in LANGS[lang].items() if k != "why"},
           "by_tier": dict(by_tier), "by_src_lang": dict(by_srclang),
           "per_source": {k: {kk: vv for kk, vv in v.most_common() if not kk.startswith("lid:")}
                          | {"top_lid_rejects": [(kk[4:], vv) for kk, vv in v.most_common() if kk.startswith("lid:")][:5]}
                          for k, v in stats.items()}}
    (RESULTS / f"S2_11_parallel_{lang}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["bug", "min", "ace", "mak"])
    a = ap.parse_args()
    for lang in a.langs:
        rep = build(lang)
        print(f"\n=== {lang}: {rep['pairs']:,} pairs  tiers {rep['by_tier']}  src {rep['by_src_lang']}")
        for k, v in rep["per_source"].items():
            print(f"    {k:55s} raw {v.get('raw',0):>9,}  kept {v.get('kept',0):>8,}  "
                  f"{v.get('kept_tgt_chars',0)/1e6:6.2f} MB  len {v.get('drop_length',0):,} laser {v.get('drop_laser',0):,} "
                  f"dupe {v.get('drop_dupe',0):,} lid {v.get('drop_lid',0):,} eval {v.get('drop_eval_overlap',0):,}")


if __name__ == "__main__":
    main()
