#!/usr/bin/env python
"""S2.13 -- NEW Indonesian task/instruction pool (seed for translation into regional languages).

Adds to dataset/midtraining/id_midtrain_balanced, never duplicates it. Everything is tagged
by provenance, because "Indonesian instruction data" hides very different things:

  aisingapore/SEA-Instruct-2602   prompts: own-synthetic / translated / open-source / human
                                  responses: Qwen3-32B generated, DeepSeek-V3.1 revised
  IndoNLU smsa emot casa wrete     native tasks, templated here
  IndoNLG indosum liputan6         native summarisation
  csebuetnlp/xlsum indonesian      native summarisation (BBC Indonesia)
  indolem/indo_story_cloze train   native commonsense
  cendol wikihow / indo_puisi      templated native text;  dolly = machine-translated
  INTISARI chat v5, jan-hq sft     synthetic, generator unknown
  cahya/instructions_indonesian, indonesian-nlp/lfqa_id   machine-translated English

DECONTAMINATION IS NOT OPTIONAL HERE. NusaX was built by translating SmSA sentences, so
SmSA (and anything quoting it) would recreate NusaX eval items once translated into a
regional language. Every row is dropped if it shares a 10-gram with NusaX Indonesian or
FLORES+ ind_Latn (dev + devtest). Benchmarks (COPAL, IndoCulture, SeaExam, M3Exam,
IndoMMLU-local) are deliberately not loaded.

    python -m scaleres.dataprep.build_id_task_pool
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from scaleres.dataprep.screen_language_corpus import ngrams, norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset/midtraining/id_task_v2"
RAW = ROOT / "dataset/raw/ind_task_src"
RESULTS = ROOT / "autoresearch/experiments/results"


def rec(source, subset, task, creation, prompt, response, flag="ok", messages=None, **extra):
    return {"lang": "ind", "source": source, "subset": subset, "task_type": task, "creation": creation,
            "quality_flag": flag, "prompt": (prompt or "").strip(), "response": (response or "").strip(),
            "messages": json.dumps(messages, ensure_ascii=False) if messages else "",
            "extra": json.dumps(extra, ensure_ascii=False)}


def fetch(url, name):
    p = RAW / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = p.with_suffix(p.suffix + ".partial")
            with open(tmp, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
            tmp.rename(p)
    return p


# ------------------------------------------------------------------ loaders
SEA_TRANSLATED = ("translated", "sailor2/")
SEA_HUMAN = ("WildChat",)


def sea_instruct():
    import pyarrow.dataset as pds
    from huggingface_hub import HfApi, hf_hub_download
    repo = "aisingapore/SEA-Instruct-2602"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("Indonesian/")]
    ds = pds.dataset([hf_hub_download(repo, f, repo_type="dataset") for f in files], format="parquet")
    cols = ["conversations", "source", "prompt_primary_domain", "prompt_primary_task", "prompt_sensitivity",
            "prompt_requires_local_cultural_knowledge", "prompt_complexity", "prompt_formality", "turn_count"]
    for b in ds.to_batches(columns=cols, batch_size=20_000):
        b = b.to_pydict()
        for i in range(len(b["source"])):
            src = b["source"][i]
            origin = ("translated" if any(k in src for k in SEA_TRANSLATED) else
                      "human" if any(k in src for k in SEA_HUMAN) else
                      "own_synthetic" if src == "aisingapore/sea-instruct" else "open_source")
            try:
                conv = ast.literal_eval(b["conversations"][i]) if isinstance(b["conversations"][i], str) else b["conversations"][i]
            except Exception:
                yield "PARSE_ERROR"; continue
            msgs = [{"role": m["role"], "content": m["content"]} for m in conv if m.get("content")]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            flag = "harmful_prompt" if str(b["prompt_sensitivity"][i]).startswith("Harmful") else "ok"
            yield rec(repo, src, str(b["prompt_primary_task"][i]), f"prompt:{origin}|response:synthetic_llm",
                      user, asst, flag=flag, messages=msgs,
                      domain=b["prompt_primary_domain"][i], sensitivity=b["prompt_sensitivity"][i],
                      local_cultural=b["prompt_requires_local_cultural_knowledge"][i],
                      complexity=b["prompt_complexity"][i], formality=b["prompt_formality"][i],
                      response_models="Qwen3-32B gen + DeepSeek-V3.1 revise")


INDONLU = {
    "smsa": ("smsa_doc-sentiment-prosa", "tsv", "Tentukan sentimen teks berikut (positive, negative, atau neutral).\n\n{t}"),
    "emot": ("emot_emotion-twitter", "csv", "Tentukan emosi yang diungkapkan dalam tweet berikut (anger, fear, happy, love, atau sadness).\n\n{t}"),
    "casa": ("casa_absa-prosa", "csv", None),
    "wrete": ("wrete_entailment-ui", "csv", None),
}


def indonlu():
    base = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset"
    for task, (d, ext, tmpl) in INDONLU.items():
        for split in ["train", "valid"]:               # test labels are masked / held out
            p = fetch(f"{base}/{d}/{split}_preprocess.{ext}", f"indonlu/{d}_{split}.{ext}")
            txt = p.read_text(encoding="utf-8")
            if ext == "tsv":
                rows = [l.rsplit("\t", 1) for l in txt.splitlines() if "\t" in l]
            else:
                rows = list(csv.reader(io.StringIO(txt)))
                header, rows = rows[0], rows[1:]
            for r in rows:
                if task == "smsa":
                    t, lab = r
                elif task == "emot":
                    lab, t = r[0], r[1]
                elif task == "casa":
                    t = r[0]
                    asp = dict(zip(header[1:], r[1:]))
                    lab = ", ".join(f"{k}: {v}" for k, v in asp.items())
                    yield rec("IndoNLP/indonlu", task, "classification", "native_task",
                              f"Tentukan sentimen untuk setiap aspek (fuel, machine, others, part, price, service) pada ulasan mobil berikut.\n\n{t}",
                              lab, split=split); continue
                else:  # wrete
                    h = dict(zip(header, r))
                    yield rec("IndoNLP/indonlu", task, "classification", "native_task",
                              f"Apakah kalimat kedua merupakan konsekuensi (entailment) dari kalimat pertama? Jawab Entail_or_Paraphrase atau NotEntail.\n\nKalimat 1: {h.get('sent_A')}\nKalimat 2: {h.get('sent_B')}",
                              h.get("label", ""), split=split); continue
                yield rec("IndoNLP/indonlu", task, "classification", "native_task", tmpl.format(t=t), lab, split=split)


def indonlg_sum():
    p = fetch("https://storage.googleapis.com/babert-pretraining/IndoNLG_finals/downstream_task/downstream_task_datasets.zip",
              "indonlg/downstream_task_datasets.zip")
    z = zipfile.ZipFile(p)
    names = z.namelist()
    for n in names:
        low = n.lower()
        if not low.endswith((".json", ".jsonl", ".csv")) or "test" in low:
            continue
        sub = "liputan6" if "liputan6" in low else "indosum" if "indosum" in low else None
        if not sub:
            continue
        raw = z.read(n).decode("utf-8", "ignore")
        try:
            items = json.loads(raw)
            items = items if isinstance(items, list) else items.get("data", [])
        except Exception:
            items = [json.loads(l) for l in raw.splitlines() if l.strip().startswith("{")]
        for it in items:
            doc = it.get("text") or it.get("document") or it.get("clean_article") or it.get("input") or ""
            summ = it.get("summary") or it.get("clean_summary") or it.get("label") or it.get("output") or ""
            doc = " ".join(doc) if isinstance(doc, list) else doc
            summ = " ".join(summ) if isinstance(summ, list) else summ
            yield rec(f"IndoNLG/{sub}", n, "summarization", "native_task",
                      f"Ringkaslah artikel berita berikut.\n\n{doc}", summ, file=n)


def xlsum():
    p = fetch("https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/indonesian_XLSum_v2.0.tar.bz2",
              "xlsum/indonesian_XLSum_v2.0.tar.bz2")
    with tarfile.open(p) as tf:
        for m in tf.getmembers():
            if not m.isfile() or "test" in m.name:
                continue
            for l in tf.extractfile(m).read().decode("utf-8").splitlines():
                it = json.loads(l)
                yield rec("csebuetnlp/xlsum", m.name, "summarization", "native_task",
                          f"Ringkaslah artikel berikut dalam satu atau dua kalimat.\n\n{it['text']}", it["summary"],
                          title=it.get("title"))


def story_cloze():
    p = fetch("https://huggingface.co/datasets/indolem/indo_story_cloze/resolve/main/train.csv", "indo_story_cloze/train.csv")
    for r in csv.DictReader(io.StringIO(p.read_text(encoding="utf-8"))):
        ctx = " ".join(r[k].strip() for k in ["Kalimat-1", "Kalimat-2", "Kalimat-3", "Kalimat-4"])
        end = r["Correct Ending"]
        yield rec("indolem/indo_story_cloze", "train", "commonsense_generation", "native_task",
                  f"Lanjutkan cerita berikut dengan satu kalimat penutup yang masuk akal.\n\n{ctx}", end)


def cendol_id():
    import pyarrow.dataset as pds
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files("indonlp/cendol_collection_v2", repo_type="dataset") if f.endswith(".parquet")]
    ds = pds.dataset([hf_hub_download("indonlp/cendol_collection_v2", f, repo_type="dataset") for f in files], format="parquet")
    tbl = ds.to_table(columns=["subset_name", "template_name", "input", "output"],
                      filter=pds.field("subset_name").isin(["wikihow", "indo_puisi", "dolly"]))
    for s, t, i, o in zip(*(tbl.column(c).to_pylist() for c in ["subset_name", "template_name", "input", "output"])):
        yield rec("indonlp/cendol_collection_v2", s, "instruction", "machine_translated" if s == "dolly" else "templated",
                  i, o, template=t)


def chat_sets():
    from datasets import load_dataset
    for repo, creation in [("INTISARI/intisari-indonesian-chat-v5", "synthetic_unknown"),
                           ("jan-hq/indonesian_sft_binarized", "unknown")]:
        for split, d in load_dataset(repo).items():
            for r in d:
                m = r["messages"]
                m = ast.literal_eval(m) if isinstance(m, str) else m
                msgs = [{"role": x["role"], "content": x["content"]} for x in m if x.get("content")]
                user = next((x["content"] for x in msgs if x["role"] == "user"), "")
                asst = next((x["content"] for x in msgs if x["role"] == "assistant"), "")
                yield rec(repo, split, "chat", creation, user, asst, messages=msgs)
    for r in load_dataset("cahya/instructions_indonesian", split="train"):
        lab = r["label"]
        if isinstance(lab, str) and lab.startswith("["):
            lab = ast.literal_eval(lab)
        if isinstance(lab, list):                       # stored as a one-element list
            lab = lab[0] if lab else ""
        if "Asisten:" in lab:
            u, a = lab.split("Asisten:", 1)
            yield rec("cahya/instructions_indonesian", "train", "instruction", "machine_translated",
                      u.replace("Pengguna:", "").strip(), a.strip())
    for r in load_dataset("indonesian-nlp/lfqa_id", split="train"):
        ans = r.get("answers") or {}
        texts = ans.get("text") if isinstance(ans, dict) else ans
        if texts:
            q = r["title"] + (("\n" + r["selftext"]) if r.get("selftext") else "")
            yield rec("indonesian-nlp/lfqa_id", "train", "long_form_qa", "machine_translated", q, texts[0])


LOADERS = [sea_instruct, indonlu, indonlg_sum, xlsum, story_cloze, cendol_id, chat_sets]


def eval_grams():
    from datasets import load_dataset
    g = set()
    for split in ["train", "validation", "test"]:
        try:
            for t in load_dataset("mteb/NusaX-senti", "ind", split=split)["text"]:
                g |= ngrams(t)
        except Exception as e:
            print("WARN nusax ind", split, e)
    for split in ["dev", "devtest"]:
        for t in load_dataset("openlanguagedata/flores_plus", "ind_Latn", split=split)["text"]:
            g |= ngrams(t)
    assert len(g) > 10_000, f"eval n-gram set suspiciously small: {len(g)}"
    return g


def existing_pool_keys():
    from datasets import load_from_disk
    d = load_from_disk(str(ROOT / "dataset/midtraining/id_midtrain_balanced"))
    return {hashlib.blake2b(norm(p or "").encode(), digest_size=8).digest() for p in d["prompt"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of loader names")
    a = ap.parse_args()
    loaders = [l for l in LOADERS if not a.only or l.__name__ in a.only]
    ev = eval_grams()
    old = existing_pool_keys()
    seen = set()
    stats = defaultdict(Counter)
    tmp = OUT.with_suffix(".jsonl.partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fo:
        for loader in loaders:
            for r in loader():
                if isinstance(r, str):
                    stats[loader.__name__][r] += 1; continue
                st = stats[f"{r['source']}|{r['creation']}"]
                st["raw"] += 1
                if len(r["prompt"]) < 5 or len(r["response"]) < 1:
                    st["drop_empty"] += 1; continue
                pk = hashlib.blake2b(norm(r["prompt"]).encode(), digest_size=8).digest()
                if pk in old:
                    st["drop_in_existing_pool"] += 1; continue
                k = hashlib.blake2b((norm(r["prompt"]) + "\t" + norm(r["response"])).encode(), digest_size=8).digest()
                if k in seen:
                    st["drop_dupe"] += 1; continue
                seen.add(k)
                if not (ngrams(r["prompt"]) | ngrams(r["response"])).isdisjoint(ev):
                    st["drop_eval_overlap"] += 1; continue
                st["kept"] += 1
                r["id"] = k.hex()
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {loader.__name__} done", flush=True)
    from datasets import Features, Value, load_dataset
    cols = ["lang", "source", "subset", "task_type", "creation", "quality_flag", "prompt", "response", "messages", "extra", "id"]
    ds = load_dataset("json", data_files=str(tmp), split="train", features=Features({c: Value("string") for c in cols}))
    ds.save_to_disk(str(OUT))
    tmp.unlink()
    rep = {"exp": "S2.13_id_task_pool", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "output": str(OUT.relative_to(ROOT)), "rows": len(ds), "eval_ngrams": len(ev),
           "by_creation": dict(Counter(ds["creation"])), "per_source": {k: dict(v) for k, v in stats.items()}}
    (RESULTS / "S2_13_id_task_pool.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"=== ind: {len(ds):,} rows")
    for k, v in rep["per_source"].items():
        print(f"    {k:75s} {dict(v)}")


if __name__ == "__main__":
    main()
