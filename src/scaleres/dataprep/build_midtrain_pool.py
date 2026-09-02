#!/usr/bin/env python
"""S2.7 -- build a deduplicated, re-weighted Indonesian midtrain pool.

The pool as shipped is not usable as-is (F18):
  * `combined_subset` is the EXACT union of the 14 sibling shards, so loading it
    alongside them doubles everything.
  * `SEACrowd_squad_id` and `local_squad_files` are byte-identical, so SQuAD-id
    appears twice inside that union.
  * After removing the duplicate, machine-translated SQuAD v2 is still 43.6% of
    the pool, and Indonesian-NATIVE content is only ~10.5%.

So the fix is not "gather more". It is: drop the duplicate, cap the dominant
translated source, and let the native sources carry the weight they deserve.

The cap is a policy, not a discovery, and it is written here rather than in a
notebook so it is arguable. `--squad-cap` limits SQuAD-id to a share of the
final pool; everything else is taken whole.

    python -m scaleres.dataprep.build_midtrain_pool --squad-cap 0.15
    python -m scaleres.dataprep.build_midtrain_pool --dry-run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "dataset/midtraining/midtraining_unified_full"
OUT = ROOT / "dataset/midtraining/id_midtrain_balanced"
REPORT = ROOT / "autoresearch/experiments/results/S2_7_pool_build.json"

UNION_SHARD = "combined_subset"
DROP = {
    "local_squad_files": "byte-identical duplicate of SEACrowd_squad_id (F18)",
    "indolem_IndoMMLU.EMPTY.bak": "the 100%-empty shard, kept only as evidence",
    UNION_SHARD: "the union of the other shards, not an independent source",
}
CAPPED = "SEACrowd_squad_id"

# Provenance, so the mixture can be reported honestly rather than by row count.
NATIVE = {
    "indolem_IndoMMLU", "afaji_indonli", "SEACrowd_indoqa",
    "SEACrowd_facqa", "nayeon212_BLEnD",
}
TRANSLATED = {"SEACrowd_squad_id", "CohereLabs_Global-MMLU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-cap", type=float, default=0.15,
                    help="max share of the final pool for the dominant "
                         "machine-translated source (0 disables it entirely)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from datasets import load_from_disk, concatenate_datasets

    shards = {}
    for d in sorted(POOL.iterdir()):
        if not d.is_dir() or d.name in DROP:
            continue
        shards[d.name] = load_from_disk(str(d))

    others = sum(len(v) for k, v in shards.items() if k != CAPPED)
    # cap*(others + n) = n  ->  n = cap*others/(1-cap)
    if a.squad_cap <= 0:
        keep_squad = 0
    elif a.squad_cap >= 1:
        keep_squad = len(shards.get(CAPPED, []))
    else:
        keep_squad = min(len(shards.get(CAPPED, [])),
                         int(a.squad_cap * others / (1 - a.squad_cap)))

    plan = {k: (keep_squad if k == CAPPED else len(v)) for k, v in shards.items()}
    total = sum(plan.values())

    print(f"dropped: {', '.join(sorted(DROP))}")
    print(f"\n{'shard':46s} {'available':>10s} {'kept':>10s} {'share':>7s}  provenance")
    for k in sorted(plan, key=lambda x: -plan[x]):
        prov = ("native" if k in NATIVE else
                "translated" if k in TRANSLATED else "synthetic/other")
        print(f"  {k[:44]:44s} {len(shards[k]):>10,} {plan[k]:>10,} "
              f"{plan[k]/total:6.1%}  {prov}")
    print(f"  {'TOTAL':44s} {'':>10s} {total:>10,}")

    by_prov = {}
    for k, n in plan.items():
        prov = ("native" if k in NATIVE else
                "translated" if k in TRANSLATED else "synthetic/other")
        by_prov[prov] = by_prov.get(prov, 0) + n
    print("\nprovenance mix:")
    for p, n in sorted(by_prov.items(), key=lambda kv: -kv[1]):
        print(f"  {p:18s} {n:>9,}  {n/total:6.1%}")

    if a.dry_run:
        return 0

    parts = []
    for k, n in plan.items():
        if n == 0:
            continue
        ds = shards[k]
        if n < len(ds):
            ds = ds.shuffle(seed=a.seed).select(range(n))
        cols = [c for c in ds.column_names]
        parts.append(ds.select_columns(sorted(cols)))
    # Align schemas. Shared column NAMES are not enough: shards disagree on the
    # TYPE of `input` and `metadata`, and concatenate_datasets fails on that
    # with an opaque "'Value' object has no attribute 'items'".
    #
    # Do NOT resolve this by dropping the offending columns. `input` is
    # Value('null') in most shards but {'context': string} in exactly the three
    # extractive-QA shards (squad_id, indoqa, facqa) -- dropping it silently
    # strips the passage those questions are about, leaving unanswerable items
    # in the pool. Flatten instead: `input_context` as a plain string, and
    # `metadata` as a JSON string so its varying keys survive.
    def normalize(ds):
        cols = set(ds.column_names)

        def fix(row):
            v = row.get("input")
            ctx = v.get("context") if isinstance(v, dict) else (v if isinstance(v, str) else None)
            md = row.get("metadata")
            # `instruction` is empty in 100% of rows in every shard -- the actual
            # prompt lives in `role_messages`, a chat list. Flatten the user
            # turns into `prompt` so the pool is usable without re-deriving this.
            msgs = row.get("role_messages") or []
            prompt = "\n\n".join(
                (m.get("content") or "") for m in msgs
                if isinstance(m, dict) and m.get("role") == "user")
            return {"input_context": ctx or "",
                    "prompt": prompt,
                    "metadata_json": json.dumps(md, ensure_ascii=False) if md else ""}

        ds = ds.map(fix, desc="flatten")
        # `translations` and `metadata` have inconsistent struct types across
        # shards; `input` is folded into input_context; `instruction` is empty
        # everywhere. role_messages is KEPT -- it is the only place the prompt
        # exists, and its type is consistent.
        drop = [c for c in ("input", "metadata", "translations", "instruction")
                if c in cols]
        return ds.remove_columns(drop)

    parts = [normalize(p) for p in parts]
    common = set(parts[0].column_names)
    for p in parts[1:]:
        common &= set(p.column_names)
    common = sorted(c for c in common
                    if len({str(p.features[c]) for p in parts}) == 1)
    parts = [p.select_columns(common) for p in parts]

    merged = concatenate_datasets(parts).shuffle(seed=a.seed)

    # Fail loud rather than shipping an unusable pool. IndoMMLU sat on disk with
    # all 14,981 rows blank for weeks because nothing checked (P1.3), and the
    # first build of THIS pool silently dropped every prompt. A midtrain corpus
    # with no prompts or no answers looks fine in a row count.
    probe = merged.select(range(min(5000, len(merged))))
    has_prompt = sum(1 for x in probe["prompt"] if (x or "").strip()) / len(probe)
    has_output = sum(1 for x in probe["output"] if (x or "").strip()) / len(probe)
    print(f"\nsanity: {has_prompt:.1%} of rows have a prompt, "
          f"{has_output:.1%} have an output")
    if has_prompt < 0.95 or has_output < 0.95:
        raise RuntimeError(
            f"refusing to write: prompt coverage {has_prompt:.1%}, output "
            f"coverage {has_output:.1%}. Both must exceed 95% -- a shard's "
            f"schema has probably changed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(OUT))
    print(f"\nwrote {len(merged):,} rows -> {OUT.relative_to(ROOT)}")
    print(f"  columns: {common}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "exp": "S2.7_pool_build",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "squad_cap": a.squad_cap, "seed": a.seed,
        "dropped": DROP, "kept_rows": plan, "total_rows": total,
        "provenance_mix": by_prov, "output": str(OUT.relative_to(ROOT)),
        "columns": common,
        "note": ("The cap is a policy choice, not a measurement. It exists "
                 "because machine-translated SQuAD was 43.6% of the "
                 "deduplicated pool and would otherwise dominate midtraining."),
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
