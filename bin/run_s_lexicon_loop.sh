#!/usr/bin/env bash
set -euo pipefail

# === Configuration ==========================================================
HF_PATHS=(
  "dataset/paralel_3_lang/topics_parallel_hf"
  "dataset/paralel_3_lang/combined_paralel_dataset_705k_dedup_clean_filtered-id"
)

LANGS=(balinese cirebonese)
OUT_DIR="metrics/s_lexicon"
OUT_PREFIX="topics_plus705k"

NUM_WORKERS="${NUM_WORKERS:-20}"
CHUNKSIZE="${CHUNKSIZE:-4}"
MAX_DOCS="${MAX_DOCS:-25000}"        # Set to empty string to disable chunking
THRESHOLD="${THRESHOLD:-0.5}"       # Set to empty string to skip quality_decision
# ============================================================================

count_docs() {
  local file=$1
  if [[ ! -f "$file" ]]; then
    echo 0
    return
  fi
  python - "$file" <<'PY'
import json, sys
path = sys.argv[1]
count = 0
seen = set()
with open(path, encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        doc_id = obj.get("doc_id")
        if isinstance(doc_id, str) and doc_id not in seen:
            seen.add(doc_id)
            count += 1
print(count)
PY
}

run_for_lang() {
  local lang=$1
  local out_json="${OUT_DIR}/${OUT_PREFIX}_${lang}.jsonl"
  local out_plot="${OUT_DIR}/${OUT_PREFIX}_${lang}.png"

  mkdir -p "$OUT_DIR"

  local before
  before=$(count_docs "$out_json")
  echo "[loop] Starting ${lang}; already have ${before} docs processed"

  local cmd=(python -m scaleres.dataprep.compute_s_lexicon --lang "$lang" --hf)
  cmd+=("${HF_PATHS[@]}" --output-doc "$out_json" --plot "$out_plot" --num-workers "$NUM_WORKERS" --chunksize "$CHUNKSIZE" --resume)
  if [[ -n "$MAX_DOCS" ]]; then
    cmd+=(--max-docs "$MAX_DOCS")
  fi
  if [[ -n "$THRESHOLD" ]]; then
    cmd+=(--threshold "$THRESHOLD")
  fi

  echo "[loop] Running: ${cmd[*]}"
  "${cmd[@]}"

  local after
  after=$(count_docs "$out_json")
  echo "[loop] Finished ${lang}; total docs now ${after}"

  if (( after <= before )); then
    COMPLETED[$lang]=1
    echo "[loop] ${lang} is complete (no new docs added)."
  fi
}

declare -A COMPLETED=()

main() {

  while (( ${#COMPLETED[@]} < ${#LANGS[@]} )); do
    for lang in "${LANGS[@]}"; do
      if [[ -n "${COMPLETED[$lang]:-}" ]]; then
        continue
      fi
      run_for_lang "$lang"
    done
  done

  echo "[loop] All languages complete."
}

main "$@"
