#!/usr/bin/env python
"""S2.7 -- audit the Indonesian midtrain/IFT pool: duplicates, composition, gaps.

Written because the pool's headline size is misleading. `combined_subset` is not
an independent sample -- it is the union of the sibling shards, and two of those
shards are byte-identical copies of machine-translated SQuAD, so the largest
component of the "Indonesian" midtrain corpus is counted twice (F18).

The output is meant to answer one question before any more data is gathered:
*what task types are actually missing*, as opposed to what is merely small.

    python -m scaleres.dataprep.audit_midtrain_pool
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POOL = ROOT / "dataset/midtraining/midtraining_unified_full"
REPORT = ROOT / "autoresearch/experiments/results/S2_7_midtrain_audit.json"

UNION_SHARD = "combined_subset"          # the union, not a source
SAMPLE = 2000                            # rows hashed per shard for identity


def shard_fingerprint(ds, n=SAMPLE) -> str:
    """Hash a strided sample of the output column -- cheap identity check."""
    col = "output" if "output" in ds.column_names else ds.column_names[-1]
    step = max(1, len(ds) // n)
    idx = list(range(0, len(ds), step))[:n]
    h = hashlib.md5()
    for i in idx:
        h.update(str(ds[i][col]).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def main():
    from datasets import load_from_disk

    shards = {}
    for d in sorted(POOL.iterdir()):
        if not d.is_dir() or d.name.endswith(".bak"):
            continue
        try:
            ds = load_from_disk(str(d))
        except Exception as ex:
            shards[d.name] = {"error": str(ex)[:200]}
            continue
        rec = {"rows": len(ds), "columns": list(ds.column_names)}
        rec["fingerprint"] = shard_fingerprint(ds)
        for f in ("task_type", "lang", "source"):
            if f in ds.column_names:
                vals = ds[f][:5000]
                rec[f] = dict(Counter(str(v) for v in vals).most_common(6))
        shards[d.name] = rec

    # duplicate detection by fingerprint + row count
    by_fp: dict[tuple, list[str]] = {}
    for name, r in shards.items():
        if "fingerprint" not in r or name == UNION_SHARD:
            continue
        by_fp.setdefault((r["fingerprint"], r["rows"]), []).append(name)
    dupes = {f"{fp}:{n}": names for (fp, n), names in by_fp.items() if len(names) > 1}

    sources = {k: v for k, v in shards.items() if k != UNION_SHARD and "rows" in v}
    total_shards = sum(v["rows"] for v in sources.values())
    union_rows = shards.get(UNION_SHARD, {}).get("rows")

    # unique rows = shard total minus one copy of every duplicate group
    dup_rows = sum(sources[names[0]]["rows"] * (len(names) - 1)
                   for names in dupes.values())
    unique = total_shards - dup_rows

    # composition after dropping duplicates
    keep = {}
    dropped = set()
    for names in dupes.values():
        for n in sorted(names)[1:]:
            dropped.add(n)
    for name, v in sources.items():
        if name not in dropped:
            keep[name] = v["rows"]

    report = {
        "exp": "S2.7_midtrain_audit",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "union_shard": UNION_SHARD,
        "union_rows": union_rows,
        "shard_total_rows": total_shards,
        "union_is_exact_sum_of_shards": union_rows == total_shards,
        "duplicate_groups": dupes,
        "duplicate_rows": dup_rows,
        "unique_rows": unique,
        "keep_composition": dict(sorted(keep.items(), key=lambda kv: -kv[1])),
        "shards": shards,
        "recommendation": (
            "Load the individual shards, never combined_subset alongside them. "
            "Drop the duplicate shard(s) listed above. Cap the largest single "
            "source before mixing -- see the share column below."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"union shard `{UNION_SHARD}`: {union_rows:,} rows")
    print(f"sum of {len(sources)} shards:  {total_shards:,} rows  "
          f"-> union is {'THE EXACT SUM' if union_rows == total_shards else 'independent'}")
    print(f"\nduplicate groups: {len(dupes)}")
    for key, names in dupes.items():
        print(f"  {sources[names[0]]['rows']:>8,} rows x{len(names)}  {names}")
    print(f"\nunique rows after dedup: {unique:,}  "
          f"({dup_rows:,} duplicated, {dup_rows/total_shards:.1%} of the pool)")

    print("\ncomposition after dropping duplicates:")
    for name, n in report["keep_composition"].items():
        bar = "#" * int(40 * n / unique)
        print(f"  {name[:44]:44s} {n:>8,}  {n/unique:6.1%} {bar}")

    print("\ntask types present (from the union shard):")
    tt = shards.get(UNION_SHARD, {}).get("task_type")
    if tt:
        for k, v in tt.items():
            print(f"  {k:28s} {v:,} (in a {SAMPLE*2.5:.0f}-row sample)")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
