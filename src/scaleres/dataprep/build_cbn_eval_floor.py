#!/usr/bin/env python
"""Carve a genuinely held-out Cirebonese eval floor, and the train set to match.

Two defects forced this (F52, F56).

  1. The old eval set (dataset/eval/cbn/ood_clean) was 93% drawn from
     cbn_valid_hq_1000, which is only ~44% Cirebonese. The resulting eval set
     was 23% pure -- mostly Indonesian -- so every Cirebonese perplexity number
     measured against it was measuring the wrong language.
  2. The clean lineage's own "val" splits are not held out. 174 of 574 rows are
     EXACT duplicates of cbn_expanded_v4_train and 252 of cbn_hq_2k; 10-gram
     overlap runs 60-98%. Decontaminating them against training leaves 10
     documents, which is not an eval set.

So the split has to be made here rather than inherited. The whole clean lineage
is pooled, language-gated, and cut into two provably disjoint halves: anything
sharing a 10-gram with a chosen eval document is removed from training.

WHICH documents become eval is chosen, not sampled. Every eval document costs
the training set all of its 10-gram neighbours, and the Cirebonese pool is only
~1.3M tokens, so candidates are ranked by how ISOLATED they are in n-gram space
and the most isolated are taken first. That buys the largest eval set for the
least training data, and the exact cost is reported rather than hidden.

s_disc gates language only -- it is chance-level on quality (F51) -- and
language is precisely what was wrong in F52.

    python -m scaleres.dataprep.build_cbn_eval_floor --target 400
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "autoresearch/exp"))
REPORT = ROOT / "autoresearch/experiments/results/cbn_eval_floor.json"

from scaleres.dataprep.build_eval_floor import ngrams, norm, NGRAM  # noqa: E402

# The only lineage that is actually Cirebonese (93-98% pure, F55).
POOL = [
    ("dataset/cpt/cbn_clean_v3_train", "cbn_clean_v3_train"),
    ("dataset/cpt/cbn_clean_v3_val", "cbn_clean_v3_val"),
    ("dataset/cpt/cbn_expanded_v4_train", "cbn_expanded_v4_train"),
    ("dataset/cpt/cbn_expanded_v4_val", "cbn_expanded_v4_val"),
]
MIN_CHARS = 60


def pct(k, n):
    return round(100.0 * k / max(1, n), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-out", default="dataset/eval/cbn/ood_clean_v2")
    ap.add_argument("--train-out", default="dataset/cpt/cbn_train_disjoint")
    ap.add_argument("--target", type=int, default=400,
                    help="eval documents to aim for")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from s_disc import s_disc_score, ANCHORS, SCORER_VERSION

    anc = ANCHORS["cirebonese"]
    rivals = {k: v for k, v in anc.items()
              if k not in ("real_target", "real_balinese")}   # ban is unscreened
    strongest = max(rivals, key=rivals.get)
    thr = a.threshold if a.threshold is not None else \
        (anc["real_target"] + rivals[strongest]) / 2
    print(f"scorer {SCORER_VERSION}")
    print(f"anchor {anc['real_target']:+.3f} | rival {strongest} "
          f"{rivals[strongest]:+.3f} | threshold {thr:+.3f}\n")

    # ---- pool, exact-deduped ------------------------------------------
    docs, seen = [], set()
    for path, name in POOL:
        d = load_from_disk(str(ROOT / path))
        added = dup = short = 0
        for t in d["text"]:
            t = (t or "").strip()
            if len(t) < MIN_CHARS:
                short += 1
                continue
            k = norm(t)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            docs.append({"text": t, "source": name})
            added += 1
        print(f"  {name:24s} rows={len(d):>6,}  +{added:<5} "
              f"(dup {dup}, short {short})")
    print(f"\npool after exact dedup: {len(docs):,}")

    # ---- language gate -------------------------------------------------
    sc = [s_disc_score(d["text"], "cirebonese") for d in docs]
    keep = [i for i, s in enumerate(sc) if s >= thr]
    print(f"language gate: {len(keep):,}/{len(docs):,} kept "
          f"({pct(len(keep), len(docs))}% pure)")

    # ---- n-gram adjacency over the gated pool --------------------------
    index = defaultdict(set)
    for i in keep:
        for g in ngrams(docs[i]["text"]):
            index[g].add(i)
    neigh = {i: set() for i in keep}
    for members in index.values():
        if len(members) > 1:
            for i in members:
                neigh[i] |= members
    for i in keep:
        neigh[i].discard(i)

    isolated = sum(1 for i in keep if not neigh[i])
    print(f"n-gram adjacency: {isolated:,} of {len(keep):,} documents are "
          f"isolated (no 10-gram shared with any other)")

    # ---- choose eval: cheapest first ----------------------------------
    # Cost of taking doc i is the neighbours it drags out of training, so take
    # isolated documents first and stop at the target.
    order = sorted(keep, key=lambda i: (len(neigh[i]), -len(docs[i]["text"])))
    chosen, banned = [], set()
    for i in order:
        if len(chosen) >= a.target:
            break
        if i in banned:
            continue
        chosen.append(i)
        banned.add(i)
        banned |= neigh[i]

    train_idx = [i for i in keep if i not in banned]
    ev_chars = sum(len(docs[i]["text"]) for i in chosen)
    tr_chars = sum(len(docs[i]["text"]) for i in train_idx)
    pool_chars = sum(len(docs[i]["text"]) for i in keep)
    cost = len(banned) - len(chosen)

    print(f"\neval  {len(chosen):>5} docs  {ev_chars/3.99/1000:>7.1f}k tokens")
    print(f"train {len(train_idx):>5} docs  {tr_chars/3.99/1000:>7.1f}k tokens")
    print(f"cost  {cost:>5} docs removed from training as 10-gram neighbours "
          f"of an eval document")
    print(f"      ({pct(pool_chars - tr_chars - ev_chars, pool_chars)}% of pool "
          f"tokens spent on disjointness)")

    # ---- verify, do not assume ----------------------------------------
    ev_grams = set()
    for i in chosen:
        ev_grams |= set(ngrams(docs[i]["text"]))
    leaks = sum(1 for i in train_idx if ev_grams & set(ngrams(docs[i]["text"])))
    print(f"\nVERIFY: train documents sharing a 10-gram with eval = {leaks} "
          f"(must be 0)")
    if leaks:
        raise SystemExit("split is not disjoint -- refusing to write")

    ev_p = pct(sum(1 for i in chosen if sc[i] >= thr), len(chosen))
    tr_p = pct(sum(1 for i in train_idx if sc[i] >= thr), len(train_idx))
    by_src = defaultdict(int)
    for i in chosen:
        by_src[docs[i]["source"]] += 1

    Dataset.from_list([{"text": docs[i]["text"], "source": docs[i]["source"],
                        "s_disc": round(sc[i], 4)} for i in chosen]
                      ).save_to_disk(str(ROOT / a.eval_out))
    Dataset.from_list([{"text": docs[i]["text"], "source": docs[i]["source"],
                        "s_disc": round(sc[i], 4)} for i in train_idx]
                      ).save_to_disk(str(ROOT / a.train_out))

    summary = {
        "exp": "cbn_eval_floor_v2", "scorer_version": SCORER_VERSION,
        "threshold": round(thr, 4), "ngram": NGRAM,
        "pool_sources": [p[1] for p in POOL],
        "pool_docs": len(docs), "pool_after_language_gate": len(keep),
        "pool_purity_pct": pct(len(keep), len(docs)),
        "isolated_docs": isolated,
        "eval_docs": len(chosen), "eval_tokens": int(ev_chars / 3.99),
        "eval_purity_pct": ev_p, "eval_by_source": dict(by_src),
        "train_docs": len(train_idx), "train_tokens": int(tr_chars / 3.99),
        "train_purity_pct": tr_p,
        "docs_sacrificed_for_disjointness": cost,
        "leak_check_train_docs_sharing_ngram": leaks,
        "eval_output": a.eval_out, "train_output": a.train_out,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    print("\n" + json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nwrote {a.eval_out}\nwrote {a.train_out}\n"
          f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
