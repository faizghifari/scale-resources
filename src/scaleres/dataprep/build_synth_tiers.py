#!/usr/bin/env python
"""Slice a raw synthetic corpus into a tier ladder instead of one filtered corpus.

For Balinese a single gate is fine: the generator has the language, so 96.6% of
output survives and the discarded remainder is genuinely bad. Cirebonese is the
opposite case -- no generator has real exposure to it, every output is partly
Javanese, and where to cut is not obvious. Cutting once forces a guess.

So cut everywhere and let a training run decide. Which tier trains best is an
empirical question no instrument here can answer in advance: six quality
instruments have failed the matched-training-run test, and s_disc among them
(F51). That makes the ladder itself the experiment -- how much filtering is
optimal for synthetic data in an extreme-scarcity language.

Thresholds are named by the fraction of known Javanese they admit rather than by
raw score, because "admits 10% of Javanese" is a statement you can defend and
tune, and "+0.716" is not.

Every tier carries its measured purity in the manifest, so a tier can never be
mistaken for real target-language text further down the pipeline.

    python -m scaleres.dataprep.build_synth_tiers \\
        --in dataset/synthetic_raw/cbn_25M --lang cirebonese \\
        --out-root dataset/cpt --prefix cbn_synth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))
REPORT = ROOT / "autoresearch/experiments/results/synth_tiers.json"

WS = re.compile(r"\s+")
WORD = re.compile(r"[a-zà-ÿ']+")
CH_PER_TOK = 3.99

# Held-out markers only -- never the forms used to steer generation, or the
# ladder would rank prompt compliance rather than language (see e2_generate).
HELD = {"cirebonese": set("pribèn priben kepriben isun reang beli gudu dulur "
                          "jeh".split()),
        "balinese": set()}
RIVAL = {"cirebonese": set("manèh maneh piyé piye kepriye saiki yèn yen waé wae "
                           "ngendi banjur dhèwè dhewe kowe".split()),
         "balinese": set()}
RIVAL_CORPUS = {"cirebonese": "dataset/cpt/jav_clean_v3",
                "balinese": "dataset/cpt/jav_clean_v3"}


def norm(t):
    return WS.sub(" ", unicodedata.normalize("NFKC", t or "")).strip().lower()


def markers(texts, lang):
    h = r = n = 0
    held, riv = HELD.get(lang, set()), RIVAL.get(lang, set())
    if not held:
        return None, None, None
    for t in texts:
        for w in WORD.findall(t.lower()):
            n += 1
            h += w in held
            r += w in riv
    f = 1000.0 / max(1, n)
    return h * f, r * f, (h / max(1, r)) if r else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--lang", required=True, choices=["cirebonese", "balinese"])
    ap.add_argument("--out-root", default="dataset/cpt")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--fpr", nargs="+", type=float, default=[0.05, 0.10, 0.20, 1.0],
                    help="1.0 = the ungated tier")
    ap.add_argument("--calib-n", type=int, default=2500)
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from s_disc import s_disc_score, SCORER_VERSION
    from scaleres.dataprep.quality_utils import evaluate_text, LANG_CONFIGS
    cfg = LANG_CONFIGS[a.lang]

    # Calibrate thresholds against the rival language, once.
    rng = random.Random(0)
    rv = load_from_disk(str(ROOT / RIVAL_CORPUS[a.lang]))
    col = "text" if "text" in rv.column_names else rv.column_names[0]
    rival_txt = [t for t in rv.select(rng.sample(range(len(rv)), a.calib_n))[col]
                 if t and len(t) > 60]
    rs = sorted(s_disc_score(t, a.lang) for t in rival_txt)
    thr = {f: (-1e9 if f >= 1.0 else rs[min(len(rs) - 1, int((1 - f) * len(rs)))])
           for f in a.fpr}
    print(f"scorer {SCORER_VERSION}")
    for f in sorted(thr):
        lab = "ungated" if f >= 1.0 else f"admits {f:.0%} of rival"
        print(f"  T({lab}) = {thr[f]:+.3f}")

    # One pass over the raw shards: quality gate, dedup, score.
    kept, seen, n_in = [], set(), 0
    drop = {"short": 0, "duplicate": 0, "degenerate": 0}
    for fn in sorted(Path(a.inp).glob("*.jsonl")):
        for line in open(fn, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_in += 1
            t = (r.get("text") or "").strip()
            if len(t) < 60:
                drop["short"] += 1; continue
            k = hashlib.blake2b(norm(t).encode(), digest_size=8).hexdigest()
            if k in seen:
                drop["duplicate"] += 1; continue
            seen.add(k)
            if not evaluate_text(t, cfg, lid=None).ok:
                drop["degenerate"] += 1; continue
            kept.append((s_disc_score(t, a.lang), t))
    print(f"\nraw {n_in:,} -> {len(kept):,} after quality gates {drop}")

    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in sorted(thr, reverse=True):
        sel = [t for s, t in kept if s >= thr[f]]
        if not sel:
            print(f"  fpr {f:.0%}: empty"); continue
        h, rv_, ratio = markers(sel, a.lang)
        ch = sum(len(t) for t in sel)
        name = "ungated" if f >= 1.0 else f"fpr{int(f*100)}"
        out = f"{a.out_root}/{a.prefix}_{name}"
        Dataset.from_dict({"text": sel}).save_to_disk(str(ROOT / out))
        row = {"tier": name, "threshold": None if f >= 1.0 else round(thr[f], 4),
               "rows": len(sel), "tokens": int(ch / CH_PER_TOK),
               "yield_pct": round(100.0 * len(sel) / max(1, n_in), 2),
               "held_marker_per_1k": None if h is None else round(h, 3),
               "rival_marker_per_1k": None if rv_ is None else round(rv_, 3),
               "held_ratio": None if ratio is None else round(ratio, 3),
               "path": out}
        manifest.append(row)
        print(f"  {name:8s} {len(sel):>8,} rows  {ch/CH_PER_TOK/1e6:>6.2f}M tok  "
              f"yield {row['yield_pct']:>5.1f}%  held-ratio "
              f"{'--' if ratio is None else f'{ratio:.2f}'}  -> {out}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"input": a.inp, "lang": a.lang, "scorer_version": SCORER_VERSION,
         "raw_docs": n_in, "after_quality": len(kept), "dropped": drop,
         "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "tiers": manifest}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
