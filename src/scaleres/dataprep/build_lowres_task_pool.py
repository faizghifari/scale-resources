#!/usr/bin/env python
"""S2.12 -- midtraining / instruction-style task pool for the regional languages.

Target-language task data barely exists, so this keeps EVERYTHING usable but tags it,
because the sources differ enormously in what they actually are:

  creation          human | templated | reverse_instruction | machine_translated
  quality_flag      ok | low_wiki_reverse | mt_nllb_needs_check

  prosa-text/nusa-dialogue   ban bug min   human dialogues + human paragraph (2 tasks each)
  indonlp/cendol_collection_v2  ban min ace bug mak   templated (Wikipedia t2t, NusaParagraph labels)
  akoksal/muri-it-language-split ban ace bug  reverse instructions over Wikipedia (cf. F15)
  alexandrainst/multi-wiki-qa  ban min ace bug  extractive QA over Wikipedia
  CohereLabs/aya_collection_language_split  ace min  NLLB-translated English sets ->
                               written to a SEPARATE pool, flagged mt_nllb_needs_check

Hard exclusions: xP3x (FLORES-only for these languages), anything NusaX/FLORES/SIB/
Belebele-derived (Aya "NusaX-senti-inst" included), our own MCQA eval (bali_mmlu).
Every row is 10-gram decontaminated against dataset/eval/<lang>/* (+ bali_mmlu for ban).

    python -m scaleres.dataprep.build_lowres_task_pool --langs ban min ace bug mak
    python -m scaleres.dataprep.build_lowres_task_pool --aya --langs ace min
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scaleres.dataprep.screen_language_corpus import eval_ngrams, ngrams, norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset/midtraining"
RESULTS = ROOT / "autoresearch/experiments/results"
LANG_NAME = {"ban": "Bali", "min": "Minangkabau", "ace": "Aceh", "bug": "Bugis", "mak": "Makassar"}


def rec(lang, source, subset, task, creation, prompt, response, flag="ok", **extra):
    return {"lang": lang, "source": source, "subset": subset, "task_type": task, "creation": creation,
            "quality_flag": flag, "prompt": prompt.strip(), "response": response.strip(),
            "extra": json.dumps(extra, ensure_ascii=False)}


# ------------------------------------------------------------------ loaders
def nusadialogue(lang):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    if lang not in ("ban", "bug", "min"):
        return
    df = pd.read_csv(hf_hub_download("prosa-text/nusa-dialogue", f"raws/{lang}.csv", repo_type="dataset"),
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    starts = df.index[df["Annotator"].astype(str).str.startswith("Annotator")].tolist() + [len(df)]
    name = LANG_NAME[lang]
    for a, b in zip(starts, starts[1:]):
        blk = df.iloc[a:b]
        head = blk.iloc[0]
        turns = [(str(s).strip(), str(c).strip()) for s, c in zip(blk["Dialogue"], blk["Unnamed: 8"])
                 if isinstance(c, str) and c.strip() and str(s).strip().lower() != "speaker"]
        para, ptype = head.get("Paragraph"), head.get("Type")
        if len(turns) < 2 or not isinstance(para, str) or not para.strip():
            continue
        dialog = "\n".join(f"{s}: {c}" for s, c in turns)
        meta = {"dialect": head.get("Dialect") if isinstance(head.get("Dialect"), str) else None,
                "topic": str(head.get("Topic")).split("\n")[0], "paragraph_type": ptype,
                "annotator": head.get("Annotator"), "template_lang": "ind"}
        yield rec(lang, "prosa-text/nusa-dialogue", "dialogue_to_paragraph", "generation", "human",
                  f"Tuliskan sebuah paragraf {ptype} dalam bahasa {name} berdasarkan dialog berikut.\n\n{dialog}",
                  para, **meta)
        yield rec(lang, "prosa-text/nusa-dialogue", "paragraph_to_dialogue", "generation", "human",
                  f"Buatlah dialog dalam bahasa {name} antara dua orang berdasarkan paragraf berikut.\n\n{para}",
                  dialog, **meta)


def cendol(lang):
    import pyarrow.dataset as pds
    from huggingface_hub import HfApi, hf_hub_download
    subs = {"ban": ["wikipedia_ban"], "ace": ["wikipedia_ace"], "bug": ["wikipedia_bug"], "min": ["wikipedia_min"],
            "mak": []}[lang]
    code = {"ban": "ban", "min": "min", "ace": "ace", "bug": "bug", "mak": "mak"}[lang]
    subs += [f"nusaparagraph_{t}_{code}_nusantara_text" for t in ("topic", "emot", "rhetoric")]
    files = [f for f in HfApi().list_repo_files("indonlp/cendol_collection_v2", repo_type="dataset") if f.endswith(".parquet")]
    paths = [hf_hub_download("indonlp/cendol_collection_v2", f, repo_type="dataset") for f in files]
    ds = pds.dataset(paths, format="parquet")
    tbl = ds.to_table(columns=["subset_name", "template_name", "input", "output"],
                      filter=pds.field("subset_name").isin(subs))
    for s, t, i, o in zip(*(tbl.column(c).to_pylist() for c in ["subset_name", "template_name", "input", "output"])):
        task = "classification" if s.startswith("nusaparagraph") else "wiki_t2t"
        yield rec(lang, "indonlp/cendol_collection_v2", s, task, "templated", i or "", o or "", template=t)


def muri(lang):
    from datasets import load_dataset
    if lang not in ("ban", "ace", "bug"):
        return
    d = load_dataset("akoksal/muri-it-language-split", lang)
    for split in d:
        for r in d[split]:
            yield rec(lang, "akoksal/muri-it-language-split", r.get("subdataset_name") or "", "instruction",
                      "reverse_instruction", r["input"], r["output"], flag="low_wiki_reverse", split=split)


def multiwikiqa(lang):
    from datasets import load_dataset
    if lang not in ("ban", "ace", "bug", "min"):
        return
    d = load_dataset("alexandrainst/multi-wiki-qa", lang)
    for split in d:
        for r in d[split]:
            ans = r.get("answers")
            ans = (ans.get("text") or [""])[0] if isinstance(ans, dict) else (ans[0] if isinstance(ans, list) and ans else ans)
            if not ans:
                continue
            # questions are LLM-generated over Wikipedia articles, not human-written
            yield rec(lang, "alexandrainst/multi-wiki-qa", "extractive_qa", "qa", "llm_generated",
                      f"{r['context']}\n\nPertanyaan: {r['question']}", str(ans), split=split,
                      template_lang="ind")


def aya(lang):
    import pyarrow.dataset as pds
    from huggingface_hub import HfApi, hf_hub_download
    cfg = {"ace": "achinese", "min": "minangkabau"}[lang]
    files = [f for f in HfApi().list_repo_files("CohereLabs/aya_collection_language_split", repo_type="dataset")
             if f.startswith(cfg + "/") and f.endswith(".parquet")]
    ds = pds.dataset([hf_hub_download("CohereLabs/aya_collection_language_split", f, repo_type="dataset")
                      for f in files], format="parquet")
    cols = [c for c in ["inputs", "targets", "dataset_name", "sub_dataset_name", "task_type", "script"]
            if c in ds.schema.names]
    for batch in ds.to_batches(columns=cols, batch_size=50_000):
        b = batch.to_pydict()
        for i in range(len(b["inputs"])):
            dn = b["dataset_name"][i] or ""
            if "nusax" in dn.lower():
                yield "EXCLUDED_NUSAX"; continue
            if "script" in b and b["script"][i] and b["script"][i] != "Latn":
                yield "EXCLUDED_NONLATN"; continue
            yield rec(lang, "CohereLabs/aya_collection_language_split", dn, b["task_type"][i] if "task_type" in b else "",
                      "machine_translated", b["inputs"][i] or "", b["targets"][i] or "", flag="mt_nllb_needs_check",
                      sub_dataset=b["sub_dataset_name"][i] if "sub_dataset_name" in b else None)
    return


MAIN = [nusadialogue, cendol, muri, multiwikiqa]


def eval_grams(lang):
    g = eval_ngrams(lang) if (ROOT / "dataset/eval" / lang).exists() else set()
    if lang == "ban":                      # the Balinese MCQA eval is not under dataset/eval
        from datasets import load_from_disk
        for q in load_from_disk(str(ROOT / "dataset/raw/ban/bali_mmlu"))["question"]:
            g |= ngrams(q)
    return g


def build(lang, loaders, out_name, tag):
    ev = eval_grams(lang)
    seen = set()
    stats = defaultdict(Counter)
    out_dir = OUT / out_name / lang
    tmp = OUT / out_name / f"{lang}.jsonl.partial"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fo:
        for loader in loaders:
            for r in loader(lang) or []:
                if isinstance(r, str):
                    stats[loader.__name__][r] += 1; continue
                st = stats[f"{r['source']}|{r['subset']}"]
                st["raw"] += 1
                if len(r["prompt"]) < 5 or len(r["response"]) < 2:
                    st["drop_empty"] += 1; continue
                k = hashlib.blake2b((norm(r["prompt"]) + "\t" + norm(r["response"])).encode(), digest_size=8).digest()
                if k in seen:
                    st["drop_dupe"] += 1; continue
                seen.add(k)
                if ev and not (ngrams(r["prompt"]) | ngrams(r["response"])).isdisjoint(ev):
                    st["drop_eval_overlap"] += 1; continue
                st["kept"] += 1
                st["kept_chars"] += len(r["prompt"]) + len(r["response"])
                r["id"] = k.hex()
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {lang} {loader.__name__} done", flush=True)
    from datasets import Features, Value, load_dataset
    feats = Features({k: Value("string") for k in
                      ["lang", "source", "subset", "task_type", "creation", "quality_flag", "prompt", "response", "extra", "id"]})
    ds = load_dataset("json", data_files=str(tmp), split="train", features=feats)
    ds.save_to_disk(str(out_dir))
    n = len(ds)
    tmp.unlink()
    by = Counter(zip(ds["creation"], ds["quality_flag"]))
    rep = {"exp": f"S2.12_{tag}", "lang": lang, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "output": str(out_dir.relative_to(ROOT)), "rows": n, "eval_ngrams": len(ev),
           "by_creation_flag": {f"{a}|{b}": v for (a, b), v in by.items()},
           "per_source": {k: dict(v) for k, v in stats.items()}}
    (RESULTS / f"S2_12_{tag}_{lang}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"=== {lang}: {n:,} rows  {rep['by_creation_flag']}")
    for k, v in rep["per_source"].items():
        print(f"    {k:70s} {dict(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ban", "min", "ace", "bug", "mak"])
    ap.add_argument("--aya", action="store_true", help="build the separate Aya MT pool instead")
    a = ap.parse_args()
    for lang in a.langs:
        if a.aya:
            build(lang, [aya], "lowres_aya_mt_unchecked", "aya_mt")
        else:
            build(lang, MAIN, "lowres_task_v1", "task_pool")


if __name__ == "__main__":
    main()
