#!/usr/bin/env python
"""Mixtures that hold the REAL FRACTION constant while total tokens scale.

F70 found that scaling synthetic data degrades out-of-distribution perplexity:
189.9 -> 205.1 as synthetic went 6.7M -> 93M. F71 found why, and it is not the
synthetic data being bad. It is lexically thin -- ~30% fewer word types per
token than real, covering 72% of the eval vocabulary at 8M tokens where real
covers 88% at 1M -- and the real half was held FIXED at 14.35M, so scaling
synthetic dropped the real share of the mixture from 68% to 13%. Perplexity
tracked the real fraction, not the synthetic volume.

If that reading is right, the fix is the mixture rather than the volume: hold
the real fraction and scale both halves by REPEATING real, and the degradation
should not appear.

The decisive cell is the largest one. F70's 107M-token mixture was 13.4% real
and scored 205.1; this builds a 107M-token mixture that is 68% real. Same total
tokens, same compute, same corpora -- only the ratio differs. If it lands near
189.9 the dilution account is confirmed; if it still degrades, the problem is
synthetic volume as such and F71 is wrong.

Upsampling repeats real documents, which risks memorisation, so the repeat
factor is reported per cell and should be read alongside the result.

    python -m scaleres.dataprep.build_upsampled_mix \\
        --real dataset/cpt/ban_clean_v6_train \\
        --synth dataset/synthetic_clean/ban_100M_real2/clean.jsonl \\
        --totals 21 42 64 107 --real-frac 0.68 --prefix ban_upmix
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CH_PER_TOK = 3.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True,
                    help="clean.jsonl, or an HF dataset dir")
    ap.add_argument("--totals", nargs="+", type=float, required=True,
                    help="total token budgets in MILLIONS")
    ap.add_argument("--real-frac", type=float, default=0.68)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-root", default="dataset/cpt")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk

    real = load_from_disk(str(ROOT / a.real)).select_columns(["text"])["text"]
    real_ch = sum(len(t) for t in real)
    real_tok = real_ch / CH_PER_TOK
    print(f"real   {len(real):>8,} docs  {real_tok/1e6:6.2f}M tokens")

    sp = ROOT / a.synth
    if sp.is_dir():
        syn = load_from_disk(str(sp)).select_columns(["text"])["text"]
    else:
        syn = []
        for line in open(sp, encoding="utf-8"):
            t = (json.loads(line).get("text") or "").strip()
            if t:
                syn.append(t)
    syn_ch_total = sum(len(t) for t in syn)
    print(f"synth  {len(syn):>8,} docs  {syn_ch_total/CH_PER_TOK/1e6:6.2f}M tokens\n")

    manifest = []
    for T in sorted(a.totals):
        want_real = T * 1e6 * a.real_frac
        want_syn = T * 1e6 * (1 - a.real_frac)
        reps = want_real / real_tok
        if want_syn * CH_PER_TOK > syn_ch_total:
            print(f"  {T:g}M: SKIPPED, needs {want_syn/1e6:.1f}M synthetic but only "
                  f"{syn_ch_total/CH_PER_TOK/1e6:.1f}M available")
            continue

        # real: whole repeats plus a prefix for the fraction
        whole, frac = int(reps), reps - int(reps)
        texts = list(real) * whole + list(real[:int(frac * len(real))])
        r_ch = sum(len(t) for t in texts)

        # synthetic: a prefix, so cells nest as in F70
        s_texts, s_ch = [], 0
        target_s_ch = want_syn * CH_PER_TOK
        for t in syn:
            if s_ch >= target_s_ch:
                break
            s_texts.append(t)
            s_ch += len(t)

        mix = Dataset.from_dict({"text": texts + s_texts}).shuffle(seed=a.seed)
        out = f"{a.out_root}/{a.prefix}_{T:g}M"
        mix.save_to_disk(str(ROOT / out))
        row = {"total_M": T, "real_frac_target": a.real_frac,
               "real_frac_actual": round(r_ch / (r_ch + s_ch), 4),
               "real_tokens_M": round(r_ch / CH_PER_TOK / 1e6, 2),
               "synth_tokens_M": round(s_ch / CH_PER_TOK / 1e6, 2),
               "real_repeats": round(reps, 2), "rows": len(mix), "path": out}
        manifest.append(row)
        print(f"  {T:>5g}M  real {row['real_tokens_M']:>6.2f}M x{reps:>4.2f}  "
              f"synth {row['synth_tokens_M']:>6.2f}M  "
              f"frac {row['real_frac_actual']:.3f}  {len(mix):>8,} rows -> {out}")

    mp = ROOT / a.out_root / f"{a.prefix}_manifest.json"
    mp.write_text(json.dumps(
        {"real": a.real, "synth": a.synth, "real_frac": a.real_frac,
         "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "cells": manifest}, indent=1), encoding="utf-8")
    print(f"\nwrote {mp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
