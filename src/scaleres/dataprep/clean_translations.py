import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .quality_utils import (
    LANG_CONFIGS,
    LANG_KEY_MAP,
    LangConfig,
    GlotLanguageIdentifier,
    evaluate_text,
    iter_translation_texts,
    load_hf_dataset_texts,
    summarize_counts,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean synthetic translation corpora")
    parser.add_argument("--lang", choices=sorted(LANG_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--translations-dir", type=Path, default=Path("synthetic_data/raw/translations")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lid-model", type=Path, default=Path("models/glotlid/model.bin")
    )
    parser.add_argument(
        "--extra-dataset",
        type=Path,
        nargs="+",
        action="extend",
        dest="extra_datasets",
        default=[],
        help="Additional HF datasets cleaned alongside translations",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )
    args = parser.parse_args()

    config = LANG_CONFIGS[args.lang]
    lang_key = LANG_KEY_MAP[args.lang]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lid = GlotLanguageIdentifier(args.lid_model, config)
    show_progress = not args.no_progress

    translation_clean, translation_rejects = _clean_translations(
        args.translations_dir, lang_key, config, lid, show_progress
    )
    write_jsonl(output_dir / "translations_clean.jsonl", translation_clean)
    write_jsonl(output_dir / "translations_rejects.jsonl", translation_rejects)

    stats = summarize_counts(
        len(translation_clean) + len(translation_rejects), len(translation_clean)
    )
    print("translations", stats)
    _summarize_rejects("translations", translation_rejects)

    if args.extra_datasets:
        for dataset_path in args.extra_datasets:
            clean, rejects = _clean_dataset(dataset_path, config, lid, show_progress)
            name = dataset_path.name
            write_jsonl(output_dir / f"{name}_clean.jsonl", clean)
            write_jsonl(output_dir / f"{name}_rejects.jsonl", rejects)
            stats = summarize_counts(len(clean) + len(rejects), len(clean))
            print(name, stats)
            _summarize_rejects(name, rejects)


def _clean_translations(
    directory: Path,
    lang_key: str,
    config: LangConfig,
    lid: GlotLanguageIdentifier,
    show_progress: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    clean: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []
    iterator = iter_translation_texts(directory, lang_key)
    if show_progress:
        iterator = tqdm(iterator, desc="translations", unit="rows")  # type: ignore[arg-type]
    for file_path, uid, text, raw in iterator:
        eval_result = evaluate_text(text, config, lid)
        if eval_result.ok:
            clean.append(
                {
                    "id": uid,
                    "text": text.strip(),
                    "lang_label": eval_result.lang_label,
                    "lang_prob": eval_result.lang_prob,
                    "source": str(file_path),
                }
            )
        else:
            rejects.append(
                {
                    "id": uid,
                    "reason": eval_result.reason,
                    "lang_label": eval_result.lang_label,
                    "lang_prob": eval_result.lang_prob,
                }
            )
    return clean, rejects


def _summarize_rejects(name: str, rejects: List[Dict[str, object]]) -> None:
    if not rejects:
        print(f"{name} reject reasons: none")
        return

    counts = Counter(str(rec.get("reason") or "unknown") for rec in rejects)
    total = sum(counts.values())
    parts = [
        f"{reason}:{count} ({count / total:.2%})"
        for reason, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    print(f"{name} reject reasons: {', '.join(parts)}")


def _clean_dataset(
    dataset_path: Path,
    config: LangConfig,
    lid: GlotLanguageIdentifier,
    show_progress: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    clean: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []
    iterator = load_hf_dataset_texts(dataset_path)
    if show_progress:
        iterator = tqdm(iterator, desc=str(dataset_path.name), unit="rows")  # type: ignore[arg-type]
    for idx, text in iterator:
        eval_result = evaluate_text(text, config, lid)
        record = {
            "index": idx,
            "text": text.strip(),
            "lang_label": eval_result.lang_label,
            "lang_prob": eval_result.lang_prob,
        }
        if eval_result.ok:
            clean.append(record)
        else:
            reject = record.copy()
            reject["reason"] = eval_result.reason
            rejects.append(reject)
    return clean, rejects


if __name__ == "__main__":
    main()
