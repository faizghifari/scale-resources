#!/usr/bin/env python
"""S2.1 -- pull raw corpora for the expansion languages from the Hub.

Every (source, config) pair comes from the phase61 sweep result, never from a
hardcoded string. That matters: the sweep already resolved config names against
the live datasets-server, and the one bug it had -- a matcher that handled `.`
and `_` but not `-` -- returned a false zero for GlotCC-V1 on every language
(F16). Re-deriving names here would re-open that failure mode.

This script PULLS ONLY. It does not screen, dedup or clean. Language screening
is S2.2 and is deliberately a separate, non-delegable step: the Balinese corpus
took ~7% English jewellery spam through a single homograph (F7), and that
happened because admission and acquisition were the same pass.

    python -m scaleres.dataprep.pull_language_corpus --langs min ace bug mak
    python -m scaleres.dataprep.pull_language_corpus --langs min --dry-run

Output: dataset/raw/{lang}/{source_slug}.jsonl, one JSON object per line with
`text` plus provenance (`source`, `config`, `hf_split`, `row_index`). Provenance
is kept per row so that a later contamination finding can be traced back to the
source that introduced it, rather than invalidating the whole corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "autoresearch/experiments/results/phase61_resource_sweep.json"
OUTDIR = ROOT / "dataset/raw"
REPORT = ROOT / "autoresearch/experiments/results/S2_1_pull_report.json"

# Sources whose rows are task/instruction/eval items, not monolingual prose.
# They are still worth pulling -- they are the only supervised data these
# languages have -- but they must not land in the pretraining pile, so they go
# to a separate subdirectory. Aya alone is 1.38 GB for Minangkabau and would
# otherwise dominate a "monolingual corpus" that is mostly prompts.
TASK_SOURCES = {
    "CohereLabs/aya_collection_language_split",
    "akoksal/muri-it-language-split",
    "alexandrainst/multi-wiki-qa",
    "indonlp/NusaX-senti",
    "Davlan/sib200",
    "google/smol",
    "HuggingFaceFW/finetranslations",
}

# Column preference when a source does not use `text`.
TEXT_COLS = ("text", "content", "raw_content", "document", "article", "sentence")

# Sources where the target-language column MUST be named explicitly, because
# the row also carries text in another language. Guessing here is how English
# gets into a Balinese corpus (F7).
#
# finetranslations is the motivating case: every row has `og_full_text` (the
# target language) alongside `translated_text` (English). A "longest string
# field" heuristic picks whichever happens to be longer per row, so it silently
# yields a mixed-language corpus that still passes a document-level check
# because most rows are fine.
SOURCE_COLUMNS = {
    "HuggingFaceFW/finetranslations": {
        "text": "og_full_text",
        "aux": {"en": "translated_text", "lang_score": "og_language_score"},
    },
    # MURI-IT: `output` is the original target-language document, `input` is a
    # reverse-generated instruction. Verified on bug -- both are Buginese.
    "akoksal/muri-it-language-split": {
        "text": "output",
        "aux": {"instruction": "input", "subdataset": "subdataset_name"},
    },
    # multi-wiki-qa: `context` is a target-language Wikipedia passage. It
    # overlaps wikimedia/wikipedia by construction, so it is kept as task data
    # and must be deduped against the Wikipedia pull in S2.2.
    "alexandrainst/multi-wiki-qa": {
        "text": "context",
        "aux": {"question": "question", "title": "title"},
    },
    # Aya: `targets` is the target-language side. See `row_filter` -- the
    # Arabic-script rows of the Minangkabau split are `<unk>`-corrupted (F19).
    "CohereLabs/aya_collection_language_split": {
        "text": "targets",
        "aux": {"prompt": "inputs", "task_type": "task_type",
                "dataset_name": "dataset_name", "script": "script"},
        "row_filter": lambda r: r.get("script") == "Latn",
        "filter_reason": "script != Latn (Arabic-script rows are <unk>-corrupted, F19)",
    },
    # smol/gatitos is a bilingual LEXICON, not prose: `src` is the English
    # headword and `trgs` a list of target-language glosses. That makes it Arm C
    # scaffolding material rather than pretraining text.
    "google/smol": {
        "text": "trgs", "join_list": " ; ",
        "aux": {"en": "src", "src_lang": "sl", "tgt_lang": "tl"},
    },
}

# Sources whose canonical repo no longer loads. `datasets` 4.x dropped script-
# based datasets, so indonlp/NusaX-senti raises "Dataset scripts are no longer
# supported"; the mteb parquet mirror carries the same rows.
SOURCE_ALIAS = {
    "indonlp/NusaX-senti": "mteb/NusaX-senti",
}


class NoTextColumn(Exception):
    """Raised instead of guessing which column holds the target language."""


def slug(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def pick_text(row: dict, source: str) -> tuple[str | None, dict]:
    """Return (text, aux). Never guesses across languages -- raises instead."""
    spec = SOURCE_COLUMNS.get(source)
    if spec:
        col = spec.get("text")
        if col is None:
            raise NoTextColumn(f"{source}: no monolingual text column defined")
        v = row.get(col)
        if isinstance(v, list):
            v = (spec.get("join_list") or " ").join(str(x) for x in v if x)
        if not isinstance(v, str) or not v.strip():
            return None, {}
        aux = {k: row.get(c) for k, c in (spec.get("aux") or {}).items()
               if row.get(c) is not None}
        return v, aux

    # An empty value in a KNOWN column means skip this row, not "this source has
    # no text column". Conflating the two aborted the whole Buginese Wikipedia
    # pull on its first blank article.
    present = [c for c in TEXT_COLS if c in row]
    if present:
        for c in present:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return v, {}
        return None, {}

    # Genuinely no known column. Do NOT fall back to the longest string field:
    # on a parallel or translation source that silently admits another language.
    raise NoTextColumn(
        f"{source}: none of {TEXT_COLS} present; columns are "
        f"{sorted(row)[:9]}. Add an explicit entry to SOURCE_COLUMNS.")


def plan(langs: list[str]) -> list[dict]:
    sweep = json.loads(SWEEP.read_text())
    jobs = []
    for src, meta in sweep["sources"].items():
        for lang in langs:
            info = (meta.get("langs") or {}).get(lang) or {}
            cfg, nbytes = info.get("config"), info.get("bytes")
            if not cfg or not nbytes:
                continue                            # unmeasured or absent
            jobs.append({
                "lang": lang, "source": src, "config": cfg,
                "expect_rows": info.get("rows"), "expect_bytes": nbytes,
                "kind": "task" if src in TASK_SOURCES else "mono",
            })
    jobs.sort(key=lambda j: (j["lang"], -(j["expect_bytes"] or 0)))
    return jobs


def pull_one(job: dict, limit: int | None) -> dict:
    from datasets import load_dataset

    sub = "task" if job["kind"] == "task" else "mono"
    out = OUTDIR / job["lang"] / sub / f"{slug(job['source'])}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return {**job, "status": "skipped-exists", "path": str(out.relative_to(ROOT)),
                "bytes": out.stat().st_size}

    t0 = time.time()
    repo = SOURCE_ALIAS.get(job["source"], job["source"])
    try:
        ds = load_dataset(repo, job["config"], streaming=True)
    except Exception as ex:
        return {**job, "status": "error", "repo": repo,
                "error": f"{type(ex).__name__}: {ex}"[:300]}

    split = "train" if "train" in ds else list(ds.keys())[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.partial")

    spec = SOURCE_COLUMNS.get(job["source"]) or {}
    row_filter = spec.get("row_filter")
    n, nb, filtered = 0, 0, 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds[split]):
                if row_filter and not row_filter(row):
                    filtered += 1
                    continue
                txt, aux = pick_text(row, job["source"])
                if not txt:
                    continue
                rec = {"text": txt, "source": job["source"],
                       "config": job["config"], "hf_split": split, "row_index": i}
                if aux:
                    rec["aux"] = aux
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                nb += len(txt)
                if limit and n >= limit:
                    break
    except NoTextColumn as ex:
        tmp.unlink(missing_ok=True)
        return {**job, "status": "needs-column-map", "error": str(ex)[:300]}
    except Exception as ex:
        tmp.unlink(missing_ok=True)
        return {**job, "status": "error", "error": f"{type(ex).__name__}: {ex}"[:300]}

    tmp.rename(out)
    rec = {**job, "status": "ok", "path": str(out.relative_to(ROOT)),
           "rows": n, "text_bytes": nb, "seconds": round(time.time() - t0, 1)}
    if repo != job["source"]:
        rec["repo_used"] = repo
    if filtered:
        rec["rows_filtered_out"] = filtered
        rec["filter_reason"] = spec.get("filter_reason")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["min", "ace", "bug", "mak"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows per source (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-source", nargs="*", default=[])
    a = ap.parse_args()

    jobs = [j for j in plan(a.langs) if j["source"] not in a.skip_source]
    print(f"{len(jobs)} (source, language) pairs to pull\n")
    for j in jobs:
        print(f"  {j['lang']:4s} {j['kind']:5s} {j['source'][:42]:42s} "
              f"{j['config'][:16]:16s} ~{(j['expect_bytes'] or 0)/1e6:8.2f} MB")
    if a.dry_run:
        return 0

    results = []
    for k, j in enumerate(jobs, 1):
        print(f"\n[{k}/{len(jobs)}] {j['lang']} <- {j['source']} ({j['config']})",
              flush=True)
        r = pull_one(j, a.limit)
        results.append(r)
        if r["status"] == "ok":
            extra = ""
            if r.get("rows_filtered_out"):
                tot = r["rows"] + r["rows_filtered_out"]
                extra = (f"  [dropped {r['rows_filtered_out']:,}/{tot:,} "
                         f"= {r['rows_filtered_out']/tot:.0%}: {r.get('filter_reason')}]")
            if r.get("repo_used"):
                extra += f"  [via {r['repo_used']}]"
            print(f"    ok  {r['rows']:,} rows  {r['text_bytes']/1e6:.2f} MB text  "
                  f"{r['seconds']}s{extra}", flush=True)
        else:
            print(f"    {r['status'].upper()}: {r.get('error', r.get('path'))}", flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "exp": "S2.1_pull",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "RAW pull only -- not screened, not deduped. See S2.2 before use.",
        "langs": a.langs, "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary ===")
    for lang in a.langs:
        for kind in ("mono", "task"):
            rows = [r for r in results
                    if r["lang"] == lang and r["kind"] == kind and r["status"] == "ok"]
            if rows:
                print(f"  {lang:4s} {kind:5s} {len(rows):2d} sources  "
                      f"{sum(r['rows'] for r in rows):>9,} rows  "
                      f"{sum(r['text_bytes'] for r in rows)/1e6:8.2f} MB")
    needs = [r for r in results if r["status"] == "needs-column-map"]
    if needs:
        print(f"\n  {len(needs)} source(s) need an explicit column map "
              f"(refused to guess -- see SOURCE_COLUMNS):")
        for r in needs:
            print(f"    {r['lang']} {r['source']}: {r.get('error')}")
    errs = [r for r in results if r["status"] == "error"]
    if errs:
        print(f"\n  {len(errs)} error(s):")
        for r in errs:
            print(f"    {r['lang']} {r['source']}: {r.get('error')}")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
