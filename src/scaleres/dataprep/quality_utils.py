import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except ImportError:  # pragma: no cover - optional dependency
    Dataset = DatasetDict = None  # type: ignore
    load_from_disk = None  # type: ignore

try:
    import fasttext
except ImportError:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

import numpy as np

LANG_KEY_MAP = {
    "balinese": "balinese",
    "cirebonese": "cirebonese",
}


@dataclass
class LangConfig:
    name: str
    allowed_langs: Sequence[str]
    min_length: int = 10
    max_token_len: int = 30
    max_repeat_ratio: float = 0.45
    max_repeat_run: int = 6
    min_lang_prob: float = 0.45
    fallback_langs: Sequence[str] = field(default_factory=tuple)

    def normalized_langs(self) -> Tuple[str, ...]:
        langs = tuple({_normalize_label(lbl) for lbl in self.allowed_langs})
        if self.fallback_langs:
            langs += tuple(
                {
                    _normalize_label(lbl)
                    for lbl in self.fallback_langs
                    if _normalize_label(lbl) not in langs
                }
            )
        return langs


LANG_CONFIGS: Dict[str, LangConfig] = {
    "balinese": LangConfig(
        name="balinese",
        allowed_langs=("ban", "ban_Latn"),
        min_lang_prob=0.8,
    ),
    "cirebonese": LangConfig(
        name="cirebonese",
        # Sundanese admitted 2026-08-19 on native-speaker judgement: Cirebon sits
        # on the Javanese-Sundanese border and Cirebonese draws on both, so a
        # sun_Latn prediction can be correct rather than a failure. Supported by
        # the lexicon signal, which is independent of LID -- sun-labelled output
        # scores median s_lexicon 0.333 against 0.444 for accepted jav-labelled
        # output, far above the 0.15 floor and overlapping it heavily.
        #
        # KNOWN COST: the filter can no longer distinguish "Cirebonese with
        # Sundanese features" from "the generator drifted into plain Sundanese".
        # sun-labelled text is 76% of what was previously rejected, so this is
        # the single largest quality assumption in the Cirebonese corpus and
        # wants native-speaker spot-checking before the corpus is trusted.
        allowed_langs=("jv", "jav", "jv_Latn", "jav_Latn",
                       "su", "sun", "su_Latn", "sun_Latn"),
        # 0.7 -> 0.5 (2026-08-19). Cirebonese has no ISO code, so GlotLID can
        # only reach it through a PROXY label (Javanese). A confidence bar tuned
        # for true Javanese therefore penalises exactly the target language:
        # measured on 800 generated documents, accepted text sits at median
        # jav-probability 0.999 -- near-certain STANDARD Javanese -- while the
        # rejected jav-labelled band sits at 0.36-0.68, which is what
        # Javanese-family-but-not-Javanese should look like.
        #
        # Safe because the label check and the probability check are independent
        # (quality_utils.evaluate_text): lowering this admits only text already
        # labelled jv/jav. sun_latn, which is 76% of all rejections and a
        # genuinely different language, stays rejected at any threshold.
        #
        # 0.5 keeps a real bar -- the top label must still be a clear majority --
        # rather than 0.3, which would admit near-coin-flip jav/sun text.
        min_lang_prob=0.5,
    ),
}


def _normalize_label(label: str) -> str:
    label = label.lower()
    if label.startswith("__label__"):
        label = label[len("__label__") :]
    return label


def _glot_labels(config: LangConfig) -> List[str]:
    candidates = list(config.allowed_langs) + list(config.fallback_langs)
    labels: List[str] = []
    for cand in candidates:
        if not cand:
            continue
        if cand.startswith("__label__"):
            labels.append(cand)
        else:
            labels.append(f"__label__{cand}")
    return list(dict.fromkeys(labels))


def _softmax(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.size == 0:
        return np.zeros(0, dtype=np.float32)
    shifted = vector - np.max(vector)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom == 0.0:
        return np.zeros_like(exp)
    return exp / denom


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class CustomLID:
    def __init__(
        self,
        model_path: Path,
        languages: Optional[Sequence[str]] = None,
        mode: str = "before",
    ) -> None:
        if fasttext is None:
            raise RuntimeError("fasttext package is required but not installed")
        self.model = fasttext.load_model(str(model_path))
        self.output_matrix = self.model.get_output_matrix()
        self.labels = list(self.model.get_labels())

        if languages and isinstance(languages, Sequence):
            unique = list(dict.fromkeys(languages))
            indices = [self.labels.index(lbl) for lbl in unique if lbl in self.labels]
        else:
            indices = list(range(len(self.labels)))

        self.language_indices = indices
        self.labels = list(np.asarray(self.labels)[self.language_indices])
        self.mode = mode

    def predict(self, text: str, k: int = 1) -> Tuple[Tuple[str, ...], np.ndarray]:
        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            return tuple(), np.zeros(0, dtype=np.float32)
        if self.mode == "after":
            return self._predict_after_softmax(cleaned, k=k)
        return self._predict_before_softmax(cleaned, k=k)

    def _sentence_vector(self, text: str) -> np.ndarray:
        vec = self.model.get_sentence_vector(text)
        return np.asarray(vec, dtype=np.float32)

    def _predict_before_softmax(
        self, text: str, k: int
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        sentence_vector = self._sentence_vector(text)
        result_vector = np.dot(
            self.output_matrix[self.language_indices, :], sentence_vector
        )
        softmax = _softmax(result_vector)
        top_idx = np.argsort(softmax)[-k:][::-1]
        return tuple(self.labels[i] for i in top_idx), softmax[top_idx]

    def _predict_after_softmax(
        self, text: str, k: int
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        sentence_vector = self._sentence_vector(text)
        result_vector = np.dot(self.output_matrix, sentence_vector)
        softmax = _softmax(result_vector)
        softmax = softmax[self.language_indices]
        top_idx = np.argsort(softmax)[-k:][::-1]
        return tuple(self.labels[i] for i in top_idx), softmax[top_idx]


class GlotLanguageIdentifier:
    def __init__(
        self, model_path: Path, config: LangConfig, mode: str = "before"
    ) -> None:
        # Load all labels so we can detect when text is classified outside the
        # intended language set; acceptance is enforced in evaluate_text.
        self.model = CustomLID(model_path, languages=None, mode=mode)

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        labels, probs = self.model.predict(text, k=1)
        if len(labels) == 0:
            return None, 0.0
        return _normalize_label(labels[0]), float(probs[0])


def basic_tokenize(text: str) -> List[str]:
    tokens = text.strip().split()
    return [tok for tok in tokens if tok]


@dataclass
class TextEval:
    ok: bool
    reason: Optional[str]
    lang_label: Optional[str] = None
    lang_prob: float = 0.0


def evaluate_text(
    text: str,
    config: LangConfig,
    lid: Optional[GlotLanguageIdentifier] = None,
) -> TextEval:
    stripped = text.strip()
    if len(stripped) < config.min_length:
        return TextEval(False, "too_short")

    tokens = basic_tokenize(stripped)
    if not tokens:
        return TextEval(False, "no_tokens")

    longest = max(len(tok) for tok in tokens)
    if longest > config.max_token_len:
        return TextEval(False, "token_too_long")

    counts = Counter(tokens)
    repeated = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
    repeat_ratio = repeated / max(1, len(tokens))
    if repeat_ratio > config.max_repeat_ratio:
        return TextEval(False, "repetition_ratio")

    max_run = _max_consecutive_run(tokens)
    if max_run > config.max_repeat_run:
        return TextEval(False, "repetition_run")

    lang_label = None
    lang_prob = 0.0
    if lid is not None:
        lang_label, lang_prob = lid.predict(stripped)
        allowed = set(config.normalized_langs())
        if lang_label is None:
            return TextEval(False, "lang_unknown")
        if lang_label not in allowed or lang_prob < config.min_lang_prob:
            return TextEval(False, "lang_mismatch", lang_label, lang_prob)

    return TextEval(True, None, lang_label, lang_prob)


def _max_consecutive_run(tokens: Sequence[str]) -> int:
    run = best = 1
    for prev, curr in zip(tokens, tokens[1:]):
        if curr == prev:
            run += 1
        else:
            best = max(best, run)
            run = 1
    best = max(best, run)
    return best


def load_hf_dataset_texts(path: Path) -> Iterator[Tuple[int, str]]:
    if load_from_disk is None:
        raise RuntimeError("datasets package is required but not installed")
    data = load_from_disk(str(path))
    if hasattr(data, "values"):
        for split in data.values():  # type: ignore[call-arg]
            yield from _iter_dataset(split)
    else:
        yield from _iter_dataset(data)


def _iter_dataset(ds: Any) -> Iterator[Tuple[int, str]]:
    if not hasattr(ds, "column_names") or "text" not in ds.column_names:
        raise KeyError("Dataset must contain a 'text' column")
    for idx, item in enumerate(ds["text"]):
        if isinstance(item, str):
            yield idx, item


def read_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def iter_translation_texts(
    directory: Path, lang_key: str
) -> Iterator[Tuple[Path, str, str, Dict[str, object]]]:
    files = sorted(directory.glob("*.jsonl"))
    for file_path in files:
        for obj in read_jsonl(file_path):
            text = None
            translations = obj.get("translations") if isinstance(obj, dict) else None
            if isinstance(translations, dict):
                text = translations.get(lang_key)
            if isinstance(text, str) and text.strip():
                uid = str(obj.get("id") or obj.get("custom_id") or "")
                yield file_path, uid, text, obj


def summarize_counts(total: int, accepted: int) -> Dict[str, float]:
    rejected = total - accepted
    rate = accepted / total if total else 0.0
    return {
        "total": float(total),
        "accepted": float(accepted),
        "rejected": float(rejected),
        "accept_rate": rate,
    }
