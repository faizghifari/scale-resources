# dataset/

All training/eval data for the scale-resources pipeline. **Gitignored** (bulk data,
not source) -- this file exists only because there's nowhere else to explain the
layout. Reorganized September 2026; see the `workspace-refactor-2026-09` note in
`../synthetic_data/GENERATION_INFO.json`'s sibling docs and
`../synthetic_data/REMOVED_ARTIFACTS.json` for what was cleaned up and why.

```
dataset/
  raw/            58G   unprocessed/lightly-collected monolingual source corpora
    ban/                Balinese: per-source subsets (sib200, nllb, glot500, nusax,
                         indonmt, madlad, wiki, udhr, muri) + aggregated all_bali_hq*
                         + glotcc/hplt/wikisource (newer, not yet merged in)
    cbn/                Cirebonese: all_cbn_hq / _clean / _dedup
    ind/                Indonesian: id_hq_data + dedup/prompt-variant stages

  cpt/            47G   continued-pretraining-ready corpora. FLAT (not per-language --
                        see below), 20 entries, all confirmed live as of 2026-09-01:
    ban_hq_200k, ban_valid_hq_5000, ban_filtered_bt-85,   <- production (src/scaleres,
    cbn_hq_2k, cbn_valid_hq_500                              bin/, train.py defaults)
    bali_hq_no_val, bali_cpt_705k, ban_clean_v6_{train,val}, cbn_hq_no_val,
    cbn_cpt_705k, cbn_clean_v3_{train,val}, cbn_expanded_v4_{train,val},
    cbn_valid_hq_1000, jav_clean_v3, jav_cpt, sun_wiki_raw, id_valid_hq_5000
                        <- autoresearch-only (real-corpus cleaning pipeline,
                           phase02e->g->k->m->q; ban_clean_v6/cbn_clean_v3/jav_clean_v3
                           are each the terminal version of their own iteration chain --
                           v2 through v5 were superseded and deleted 2026-09-01)

  parallel/       45G   multi-language aligned corpora
    2lang/                id<->cbn / jv<->cbn / su<->cbn bilingual sets (early pipeline)
    3lang/                id/ban/cbn combined + annotation-filtered variants
    synthetic/            THE canonical synthetic corpus (HF DatasetDict: raw +
                          filtered_heuristic splits) -- published to
                          huggingface.co/datasets/haznitrama/idn-ban-cbn-synthetic,
                          built by scaleres.dataprep.build_parallel_hf_dataset from
                          ../synthetic_data/raw/translations. Full provenance in
                          ../synthetic_data/GENERATION_INFO.json.
    synthetic_clean/      ad hoc autoresearch derivative of synthetic/ ("clean_v1"
                          split) = Arm C in the arms/ research below

  arms/           4.2G  autoresearch's SAE-gap Arm A/B/C experiment data (see
                        arm-a-gap-guidance-dissociation memory for what these mean)
    a/{ban,cbn}/          Arm A: raw/ (irreplaceable generation output, Cirebonese
                          cost $36.85), dedup/ (deduped), pilot/ (earlier smaller
                          snapshot behind the original published A-vs-B finding)
    b/ban/                Arm B: same raw/dedup/pilot shape (no Cirebonese Arm B
                          was ever generated)
    c/                    Arm C derivatives: keep_indices/ (dedup-pass allowlists),
                          scaling_subsets/{10M,30M,100M,300M} (token-budgeted
                          curves), val_5000/ (held-out diagnostic val set). The
                          Arm C corpus itself is parallel/synthetic_clean/ above.
    pool_shards/          every historical gen_{ban,cbn}_*.jsonl run (not arm-
                          specific), merged + deduped into one pool per language

  ift/            50G   instruction-tuning data (ban_ift_6k)
  midtraining/    2.1G  QA datasets for the midtraining stage (SQuAD/TyDiQA-derived,
                        prepared by scaleres.dataprep.prepare_unified_dataset)
  eval/           136M  raw/ (quiz source material) + ban/ + cbn/ (finished
                        benchmark datasets: minimal pairs, NusaX OOD, etc.)
  annotation/     2.5M  raw/ (human QA annotation CSV sheets) + ban_sample_300 /
                        cbn_sample_300 (derived HF samples)
```

## Why `cpt/` is flat but `raw/`/`eval/`/`annotation/` are per-language or raw/-split

`cpt/`'s 20 entries were individually verified live (5 by grepping production code,
15 by cross-referencing every `autoresearch/experiments/*.py` read/write path against
each script's actual current default, not just docstring prose -- see
`../synthetic_data/REMOVED_ARTIFACTS.json`'s cpt/ entry for the full verdict list).
They were deliberately **not** nested into per-language subfolders like `raw/` --
that would have required updating ~20 actively-used autoresearch research scripts for
a purely cosmetic reorganization. If you add a `dataset/cpt/{ban,cbn,...}/` structure
later, update those scripts' path constants in the same pass.

## Scratch vs archive

`autoresearch/exp/cache/` (not under `dataset/`) is the autoresearch program's
*working scratch space* -- where its scripts write fresh output when rerun. This
`dataset/arms/` tree is the *organized archive* of the canonical/final corpora as of
the last consolidation pass (2026-09-01). A rerun regenerates into scratch and needs
its own consolidation pass into `dataset/arms/` to become canonical again -- it isn't
automatic.
