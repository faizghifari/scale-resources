#!/usr/bin/env bash
# Rebuild filtered_heuristic with current heuristics and report Balinese/Cirebonese token counts.
# Usage: bash bin/run_filter_and_counts.sh

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
DATA_DIR="$ROOT_DIR/dataset/parallel/synthetic"
LID_MODEL="$ROOT_DIR/models/glotlid/model.bin"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python at $VENV_PY" >&2
  exit 1
fi

if [[ ! -f "$LID_MODEL" ]]; then
  echo "Missing GlotLID model at $LID_MODEL" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/2] Rebuilding filtered_heuristic..."
"$VENV_PY" -m scaleres.dataprep.refresh_filtered_subset --lid-model "$LID_MODEL"

echo "[2/2] Computing token totals (Balinese, Cirebonese)..."
"$VENV_PY" - <<'PY'
from datasets import load_from_disk
import tiktoken
from pathlib import Path
from tqdm import tqdm

def get_encoding():
    try:
        return tiktoken.encoding_for_model("gpt-5-nano")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

data_path = Path('dataset/parallel/synthetic')
ds = load_from_disk(str(data_path))
filtered = ds['filtered_heuristic']
enc = get_encoding()

bal_tokens = 0
cbr_tokens = 0
for row in tqdm(filtered, total=len(filtered), desc="Tokenizing filtered", unit="rows"):
    bal_tokens += len(enc.encode(row['balinese'], disallowed_special=()))
    cbr_tokens += len(enc.encode(row['cirebonese'], disallowed_special=()))

print({
    'filtered_rows': len(filtered),
    'balinese_tokens': bal_tokens,
    'cirebonese_tokens': cbr_tokens,
})
PY

echo "Done."
