#!/usr/bin/env python
"""S2.2 -- dedup and per-segment language screening for the expansion languages.

Three passes, in this order, because each one makes the next cheaper:

  1. EXACT dedup on normalised text. wikimedia/wikipedia and
     omarkamali/wikipedia-monthly are near-copies of each other, and
     goldfish-models/fish-food is itself built from Glot500 and HPLT, so the
     overlap between sources is structural, not incidental.
  2. NEAR dedup via MinHash + LSH on 5-word shingles. Catches the same article
     with a different revision date or boilerplate.
  3. PER-SEGMENT language screening. Document-level LID is what let ~7% English
     jewellery spam into the Balinese corpus through the single word `ring`
     (F7): a mostly-Balinese page with an English block passes as Balinese, and
     the English block then trains the model.

The screening rule is a MARGIN, not a threshold. For these languages the
confusable is not noise, it is a real neighbour that shares vocabulary --
Minangkabau against Indonesian above all, where the overlap is structural rather
than accidental. So a segment is admitted only when

    p(target) - max_c p(confusable_c) >= margin

which is the same discriminative move that fixed s_lexicon in phase60 (F5/F6):
an absolute score lets the majority-shared language win, a contrast does not.

    python -m scaleres.dataprep.screen_language_corpus --langs min ace bug mak
    python -m scaleres.dataprep.screen_language_corpus --langs min --limit 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "dataset/raw"
OUT = ROOT / "dataset/clean"
GLOTLID = ROOT / "models/glotlid/model.bin"
REPORT = ROOT / "autoresearch/experiments/results/S2_2_screen_report.json"

# target label(s), then the languages it is actually confusable with.
# `screen` is what the margin is taken against; English is in every list because
# web-scraped corpora for every one of these languages carry English blocks.
LANGS = {
    "min": {"target": ["min_Latn"],
            "screen": ["ind_Latn", "msa_Latn", "zlm_Latn", "bjn_Latn", "eng_Latn"],
            "margin": 0.25,
            "why": "Indonesian overlap is structural -- Minangkabau is Malayic and "
                   "shares core vocabulary, so an absolute threshold admits Indonesian."},
    "ace": {"target": ["ace_Latn"],
            "screen": ["ind_Latn", "eng_Latn", "msa_Latn"],
            "margin": 0.15,
            "why": "No close neighbour (Chamic), so the real risk is English and "
                   "Indonesian boilerplate rather than a sister language."},
    "bug": {"target": ["bug_Latn"],
            "screen": ["ind_Latn", "mak_Latn", "eng_Latn"],
            "margin": 0.15,
            "why": "Makassarese is the sister language and the intended neighbour; "
                   "keeping them apart is what makes the neighbour cell meaningful."},
    "mak": {"target": ["mak_Latn"],
            "screen": ["ind_Latn", "bug_Latn", "eng_Latn"],
            "margin": 0.15,
            "why": "Mirror of bug."},
}

MIN_SEG_CHARS = 40        # below this GlotLID is unreliable; segment is dropped
MIN_DOC_CHARS = 100       # a document this short after screening is not worth keeping
MIN_KEEP_FRACTION = 0.5   # drop the document if under half its text survived
SHINGLE = 5
NUM_PERM = 64
BANDS = 16                # 16 bands x 4 rows -> ~0.7 Jaccard threshold

WS = re.compile(r"\s+")
PARA = re.compile(r"\n\s*\n+")


def norm(text: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()


def h64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


def minhash(text: str) -> tuple[int, ...] | None:
    words = norm(text).split()
    if len(words) < SHINGLE:
        return None
    shingles = {h64(" ".join(words[i:i + SHINGLE]).encode())
                for i in range(len(words) - SHINGLE + 1)}
    if not shingles:
        return None
    # cheap permutations: xor with a fixed salt, take the min
    return tuple(min(s ^ (i * 0x9E3779B97F4A7C15) for s in shingles)
                 for i in range(NUM_PERM))


def segments(text: str) -> list[str]:
    """Paragraph segmentation, with long paragraphs split on sentence ends."""
    out = []
    for para in PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= 1200:
            out.append(para)
            continue
        buf = ""
        for piece in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(piece) > 1000 and buf:
                out.append(buf.strip())
                buf = piece
            else:
                buf += " " + piece
        if buf.strip():
            out.append(buf.strip())
    return out


class Screener:
    def __init__(self, cfg: dict):
        import fasttext
        self.model = fasttext.load_model(str(GLOTLID))
        self.target = set(cfg["target"])
        self.screen = set(cfg["screen"])
        self.margin = cfg["margin"]

    def judge(self, seg: str) -> tuple[bool, str, float]:
        """(accept, top_label, margin). Predicts enough labels to see the rival."""
        labels, probs = self.model.predict(seg.replace("\n", " "), k=12)
        labels = [l.replace("__label__", "") for l in labels]
        p = dict(zip(labels, (float(x) for x in probs)))
        p_t = max((p.get(t, 0.0) for t in self.target), default=0.0)
        p_c = max((p.get(c, 0.0) for c in self.screen), default=0.0)
        top = labels[0] if labels else "none"
        return (p_t - p_c) >= self.margin and top in self.target, top, p_t - p_c


def run_language(lang: str, limit: int | None) -> dict:
    cfg = LANGS[lang]
    src_dir = RAW / lang / "mono"
    if not src_dir.exists():
        return {"lang": lang, "error": f"no {src_dir}"}

    sc = Screener(cfg)
    seen_exact: set[int] = set()
    bands: dict[tuple, list] = defaultdict(list)
    out_path = OUT / lang / "mono.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, Counter] = defaultdict(Counter)
    rejected_labels: Counter = Counter()
    kept_docs = kept_chars = 0
    seen_docs = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for f in sorted(src_dir.glob("*.jsonl")):
            src = f.stem
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if limit and seen_docs >= limit:
                        break
                    seen_docs += 1
                    rec = json.loads(line)
                    text = rec.get("text") or ""
                    st = per_source[src]
                    st["in"] += 1
                    st["in_chars"] += len(text)

                    if len(text) < MIN_DOC_CHARS:
                        st["drop_short"] += 1
                        continue

                    key = h64(norm(text).encode())
                    if key in seen_exact:
                        st["drop_exact_dupe"] += 1
                        continue
                    seen_exact.add(key)

                    sig = minhash(text)
                    if sig is not None:
                        rows = NUM_PERM // BANDS
                        keys = [(b,) + sig[b * rows:(b + 1) * rows] for b in range(BANDS)]
                        if any(bands[k] for k in keys):
                            st["drop_near_dupe"] += 1
                            continue
                        for k in keys:
                            bands[k].append(1)

                    keep, dropped_chars = [], 0
                    for seg in segments(text):
                        if len(seg) < MIN_SEG_CHARS:
                            dropped_chars += len(seg)
                            st["seg_too_short"] += 1
                            continue
                        ok, top, _m = sc.judge(seg)
                        if ok:
                            keep.append(seg)
                            st["seg_keep"] += 1
                        else:
                            dropped_chars += len(seg)
                            st["seg_drop"] += 1
                            rejected_labels[top] += 1

                    body = "\n\n".join(keep)
                    total = len(body) + dropped_chars
                    if not body or len(body) < MIN_DOC_CHARS:
                        st["drop_no_content"] += 1
                        continue
                    if total and len(body) / total < MIN_KEEP_FRACTION:
                        st["drop_mostly_foreign"] += 1
                        continue

                    fout.write(json.dumps(
                        {"text": body, "source": rec.get("source"),
                         "config": rec.get("config"), "row_index": rec.get("row_index")},
                        ensure_ascii=False) + "\n")
                    st["out"] += 1
                    st["out_chars"] += len(body)
                    kept_docs += 1
                    kept_chars += len(body)

    return {
        "lang": lang,
        "config": {k: v for k, v in cfg.items() if k != "why"},
        "why": cfg["why"],
        "docs_in": seen_docs, "docs_out": kept_docs,
        "chars_out": kept_chars,
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "rejected_segment_labels": dict(rejected_labels.most_common(15)),
        "output": str(out_path.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["min", "ace", "bug", "mak"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max input docs per language (smoke tests)")
    a = ap.parse_args()

    results = []
    for lang in a.langs:
        print(f"\n=== {lang} ===", flush=True)
        r = run_language(lang, a.limit)
        results.append(r)
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  {r['docs_in']:,} docs in -> {r['docs_out']:,} out "
              f"({r['chars_out']/1e6:.2f} MB)")
        for src, st in sorted(r["per_source"].items(),
                              key=lambda kv: -kv[1].get("out_chars", 0)):
            ic, oc = st.get("in_chars", 0), st.get("out_chars", 0)
            print(f"    {src[:40]:40s} {st.get('in',0):>9,} -> {st.get('out',0):>9,} docs   "
                  f"{ic/1e6:8.2f} -> {oc/1e6:7.2f} MB  "
                  f"({oc/ic if ic else 0:5.1%} kept)")
        print(f"    top rejected segment labels: "
              f"{list(r['rejected_segment_labels'].items())[:6]}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "exp": "S2.2_screen",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Per-segment LID with a margin against named confusables, after "
                 "exact and MinHash near-dedup. Segment counts are segments, not docs."),
        "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
