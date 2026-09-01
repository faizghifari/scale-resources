# scale-resources

Balinese/Cirebonese low-resource-language scaling pipeline: collecting real corpora,
generating synthetic Indonesian→Balinese/Cirebonese parallel data, and training/
evaluating small language models on the result. Indonesian is used throughout as the
high-resource neighbor language; Javanese and Sundanese show up as related-language
reference/comparison corpora.

This repo also hosts a separate autonomous research program (SAE coverage-gap-guided
synthetic data) under `autoresearch/` -- see `autoresearch/README.md`. It shares this
repo's `dict/` lexicons and reads some `dataset/` corpora, but is otherwise
self-contained and out of scope for the rest of this document.

## Layout

```
src/scaleres/          installable package (pip install -e .)
  generation/           topic/question generation, answer generation, translation
  dataprep/             build/clean/filter/push the parallel HF dataset
  training/             CPT/tokenizer/SFT/LoRA-merge training scripts
  eval/                 MCQA and perplexity evaluation
  common/                shared: LLM client, JSONL resumable I/O, HF dataset helpers,
                          lexicon lookup, Unsloth LoRA helpers
bin/                    shell wrappers around `python -m scaleres.<pkg>.<mod>`
dict/                   Balinese/Cirebonese <-> Indonesian lexicons + s_lexicon weights
synthetic_data/         synthetic-generation pipeline's seeds + raw per-topic shards
                        (see synthetic_data/GENERATION_INFO.json for full provenance)
dataset/                all training/eval data -- gitignored, see dataset/README.md
notebooks/legacy/       archived one-off notebooks, superseded by src/scaleres/
models/                 trained checkpoints, tokenizers, LID models -- gitignored
autoresearch/           separate SAE-gap-guided synthetic data research program
```

`dataset/`, `models/`, `metrics/`, and `autoresearch/` are all gitignored (bulk data /
experiment artifacts, not source). Only code, lexicons, and pipeline-provenance JSON
are tracked.

## Pipeline overview

1. **Topic generation** (`scaleres.generation.generate_topics`) -- seed topics -> LLM
   expansion into subtopics/questions.
2. **Answer generation** (`scaleres.generation.generate_synthetic_answers`) -- vLLM
   answers each question in Indonesian, resumable per-topic JSONL output.
3. **Translation** (`scaleres.generation.translate_answers` /
   `run_translate_chunks`) -- vLLM translates each answer into Balinese and
   Cirebonese, with lexicon hints from `dict/`.
4. **Packing** (`scaleres.dataprep.build_parallel_hf_dataset`) -- per-topic JSONL ->
   a single HF `DatasetDict` (`dataset/parallel/synthetic`).
5. **Filtering** (`scaleres.dataprep.refresh_filtered_subset`,
   `clean_translations`) -- language-ID + repetition/length heuristics.
6. **Publish** (`scaleres.dataprep.push_hf_parallel_dataset`) -- pushes to
   `huggingface.co/datasets/haznitrama/idn-ban-cbn-synthetic`.
7. **Training** (`scaleres.training.train` / `train_tokenizer` / `cpt_unsloth` /
   `sft_unsloth`) -- from-scratch or LoRA CPT/SFT on `dataset/cpt/` and
   `dataset/ift/`.
8. **Evaluation** (`scaleres.eval.eval_mcqa` / `eval_ppl`) -- MCQA and perplexity
   against `dataset/eval/`.

Full provenance for the synthetic-data pipeline specifically (models, prompts, exact
commands, what's live vs deleted-as-redundant) is in
`synthetic_data/GENERATION_INFO.json`; a running log of every dataset/code artifact
removed as redundant during the Sept 2026 workspace refactor, with the evidence for
each call, is in `synthetic_data/REMOVED_ARTIFACTS.json`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Unsloth LoRA training needs a GPU-matched build, installed separately:
# pip install -e ".[unsloth]"
```

Copy `.env` with API keys as needed (OpenRouter/OpenAI for generation, HF_TOKEN for
publishing datasets/models).
