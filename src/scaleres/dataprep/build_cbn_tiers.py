#!/usr/bin/env python
"""Tier the disputed Cirebonese corpora instead of accepting or discarding them.

Cirebonese data is scarce enough that a binary keep/discard verdict wastes real
signal (F60). The first cleanup pass thresholded each corpus as a whole and
called everything under a 35% keep rate unsalvageable -- but two of those
corpora have a small genuine Cirebonese core, and throwing the corpus away threw
that away too.

So the threshold is set by FALSE-POSITIVE RATE against known Javanese rather
than by a midpoint between anchors. "Admit at most 10% of Javanese" is a
statement you can defend and tune; "+0.775" is not.

Every tier is verified with an instrument INDEPENDENT of the score that built
it: rates of Cerbon-Dermayu forms (maning, priben, sekien, lamon, kuen, isun)
against their standard Javanese counterparts (maneh, piye, saiki, yen, wae).
Real Cirebonese sits at ratio ~16.7, real Javanese at ~0.01. A tier whose
survivors do not beat the Javanese ratio is not Cirebonese, whatever it scored.

    python -m scaleres.dataprep.build_cbn_tiers --fpr 0.05 0.10

CAVEAT the report repeats: the marker test is uninformative on krama-saturated
corpora (id_cbn_127k, jv_cbn_24k), because the marker lists are ngoko forms
that would be absent from krama text regardless of language.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))
REPORT = ROOT / "autoresearch/experiments/results/cbn_tiers.json"

W = re.compile(r"[a-zà-ÿ']+")
CBN_MARK = set("maning pribèn priben kepriben sekien lamon kuen isun reang beli "
               "gudu baé bae dulur jeh".split())
JAV_MARK = set("manèh maneh piyé piye kepriye saiki yèn yen waé wae ngendi "
               "banjur dhèwè dhewe kowe".split())
KRAMA = set("ingkang saged utawi sanèsipun meniko menika punika sampun dhateng "
            "dumateng wonten".split())

DISPUTED = [
    ("dataset/parallel/3lang/cirebonese_annotation-filter_bt-valid-pct_80", "cirebonese"),
    ("dataset/parallel/3lang/cirebonese_annotation-filter_valid-pct_50", "cirebonese"),
    ("dataset/parallel/3lang/combined_705k_dedup_clean_filtered-id", "cirebonese"),
    ("dataset/parallel/3lang/filtered_1k_short_nowiki", "cirebonese"),
    ("dataset/parallel/2lang/id_cbn_127k_filtered", "translated_text"),
    ("dataset/parallel/2lang/jv_cbn_24k", "translated_text"),
    ("dataset/parallel/2lang/su_cbn_28k", "translated_text"),
    ("dataset/parallel/synthetic_clean/clean_v1", "cirebonese"),
]
KRAMA_UNINFORMATIVE = {"dataset/parallel/2lang/id_cbn_127k_filtered",
                       "dataset/parallel/2lang/jv_cbn_24k"}
MIN_CHARS = 60


def marker_rates(texts):
    c = j = k = n = 0
    for t in texts:
        for w in W.findall((t or "").lower()):
            n += 1
            c += w in CBN_MARK
            j += w in JAV_MARK
            k += w in KRAMA
    f = 1000.0 / max(1, n)
    return c * f, j * f, k * f


def load_col(path, col, rng, cap=0):
    from datasets import load_from_disk
    d = load_from_disk(str(ROOT / path))
    if hasattr(d, "keys") and not hasattr(d, "column_names"):
        d = d[list(d.keys())[0]]
    idx = list(range(len(d)))
    if cap and cap < len(d):
        idx = sorted(rng.sample(idx, cap))
    view = d.select(idx)
    return view, [t or "" for t in view[col]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", nargs="+", type=float, default=[0.05, 0.10])
    ap.add_argument("--cap", type=int, default=0,
                    help="rows per corpus (0 = all); calibration always samples")
    ap.add_argument("--min-ratio", type=float, default=1.0,
                    help="a tier must beat this cbn:jav marker ratio to be written")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from s_disc import s_disc_score, SCORER_VERSION
    rng = random.Random(0)

    # Calibrate thresholds on known Javanese: T(f) admits at most f of it.
    _, jav = load_col("dataset/cpt/jav_clean_v3", "text", rng, cap=3000)
    jav = [t for t in jav if len(t) > MIN_CHARS]
    js = sorted(s_disc_score(t, "cirebonese") for t in jav)
    thr = {f: js[min(len(js) - 1, int((1 - f) * len(js)))] for f in a.fpr}

    _, real = load_col("dataset/eval/cbn/ood_clean_v2", "text", rng)
    rs = [s_disc_score(t, "cirebonese") for t in real]
    rc, rj, rk = marker_rates(real)
    jc, jj, jk = marker_rates(jav)

    print(f"scorer {SCORER_VERSION}")
    print(f"reference  real cbn  cbn/1k {rc:5.2f}  jav/1k {rj:5.2f}  "
          f"ratio {rc/max(.01,rj):6.2f}  krama/1k {rk:5.2f}")
    print(f"reference  real jav  cbn/1k {jc:5.2f}  jav/1k {jj:5.2f}  "
          f"ratio {jc/max(.01,jj):6.2f}  krama/1k {jk:5.2f}\n")
    for f in sorted(thr):
        tpr = 100.0 * sum(1 for s in rs if s >= thr[f]) / len(rs)
        print(f"  T(fpr={f:.0%}) = {thr[f]:+.3f}  keeps {tpr:.1f}% of real Cirebonese")
    print()

    hdr = (f"{'corpus':52s} {'tier':>10} {'kept':>8} {'%':>6} "
           f"{'cbn/1k':>7} {'jav/1k':>7} {'ratio':>6}  verdict")
    print(hdr); print("-" * len(hdr))

    results = []
    for path, col in DISPUTED:
        if not (ROOT / path).exists():
            print(f"{path:52s} MISSING"); continue
        view, texts = load_col(path, col, rng, cap=a.cap)
        sc = [s_disc_score(t, "cirebonese") if len(t) > MIN_CHARS else -9.0
              for t in texts]
        for f in sorted(thr):
            keep = [i for i, s in enumerate(sc) if s >= thr[f]]
            if not keep:
                print(f"{path[-52:]:52s} {f:>9.0%} {0:>8} {0.0:>5.1f}% "
                      f"{'--':>7} {'--':>7} {'--':>6}  empty")
                continue
            c, j, k = marker_rates([texts[i] for i in keep])
            ratio = c / max(0.01, j)
            uninf = path in KRAMA_UNINFORMATIVE
            if uninf:
                verdict = "markers uninformative (krama)"
                ok = False
            elif ratio >= a.min_ratio:
                verdict = "TIER B -- cirebonese-leaning"
                ok = True
            else:
                verdict = "TIER C -- javanese-grade"
                ok = False
            out = None
            if ok and not a.dry_run:
                out = f"{path}_tierB_fpr{int(f*100)}"
                view.select(keep).save_to_disk(str(ROOT / out))
                verdict += f" -> {Path(out).name}"
            print(f"{path[-52:]:52s} {f:>9.0%} {len(keep):>8,} "
                  f"{100*len(keep)/len(texts):>5.1f}% {c:>7.2f} {j:>7.2f} "
                  f"{ratio:>6.2f}  {verdict}")
            results.append({
                "path": path, "column": col, "fpr": f,
                "threshold": round(thr[f], 4), "rows_in": len(texts),
                "rows_kept": len(keep), "cbn_per_1k": round(c, 3),
                "jav_per_1k": round(j, 3), "krama_per_1k": round(k, 3),
                "marker_ratio": round(ratio, 3),
                "markers_uninformative": uninf,
                "output": out, "verdict": verdict})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "scorer_version": SCORER_VERSION,
        "reference_real_cbn": {"cbn_per_1k": round(rc, 3), "jav_per_1k": round(rj, 3),
                               "ratio": round(rc / max(.01, rj), 3)},
        "reference_real_jav": {"cbn_per_1k": round(jc, 3), "jav_per_1k": round(jj, 3),
                               "ratio": round(jc / max(.01, jj), 3)},
        "thresholds": {str(k): round(v, 4) for k, v in thr.items()},
        "min_ratio": a.min_ratio,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
