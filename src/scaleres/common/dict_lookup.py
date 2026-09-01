"""Load Indonesian<->target-language lexicon JSON files (``dict/*.json``) and
build inline lexicon hint blocks for translation prompts.

This is deliberately separate from ``scaleres.dataprep.compute_s_lexicon``'s
lexicon handling, which builds a root/gloss index for S_lexicon quality
*scoring* -- a different algorithm solving a different problem than the
prompt-hinting done here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

STRIP_CHARS = "\"'()[]{}.,;:!?<>«»“”‘’—-"


def load_dictionary(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k).lower(): [str(vv) for vv in value] for k, value in data.items()}


def tokenize_for_lexicon(text: str) -> List[str]:
    raw_tokens = text.lower().split()
    tokens: List[str] = []
    for token in raw_tokens:
        cleaned = token.strip(STRIP_CHARS)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    for idx in range(len(tokens) - n + 1):
        yield " ".join(tokens[idx : idx + n])


def build_lexicon_block(text: str, dictionary: Dict[str, List[str]]) -> str:
    tokens = tokenize_for_lexicon(text)
    seen: Set[str] = set()
    lines: List[str] = []
    for n in (3, 2, 1):
        for ngram in iter_ngrams(tokens, n):
            if ngram in seen:
                continue
            translations = dictionary.get(ngram)
            if translations:
                seen.add(ngram)
                joined = ", ".join(translations)
                lines.append(f"- {ngram}: {joined}")
    return "\n".join(lines) if lines else "None"
