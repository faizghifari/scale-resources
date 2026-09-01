import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set, cast

from tqdm import tqdm
from datasets import Dataset, Features, Value

from .quality_utils import ensure_directory, read_jsonl


def load_clean_ids(path: Optional[Path], label: str) -> Optional[Set[str]]:
    if path is None:
        return None
    if not path.exists():
        print(
            f"[build_parallel_hf_dataset] Warning: {label} clean file {path} not found; skipping cleaning",
            flush=True,
        )
        return None
    ids: Set[str] = set()
    for obj in read_jsonl(path):
        if not isinstance(obj, dict):
            continue
        doc_id = obj.get("id") or obj.get("custom_id")
        if isinstance(doc_id, str) and doc_id.strip():
            ids.add(doc_id.strip())
    print(
        f"[build_parallel_hf_dataset] Loaded {len(ids)} cleaned IDs for {label} from {path}",
        flush=True,
    )
    return ids


def iter_parallel_records(
    topic_dir: Path,
    balinese_ids: Optional[Set[str]] = None,
    cirebonese_ids: Optional[Set[str]] = None,
    show_progress: bool = True,
) -> Callable[[], Iterable[Dict[str, str]]]:
    files = sorted(topic_dir.glob("topic*.jsonl"))
    total_files = len(files)

    def generator() -> Iterable[Dict[str, str]]:
        seen: Set[str] = set()
        file_iter = files
        if show_progress:
            file_iter = tqdm(files, desc="topics", unit="files")

        for idx, file_path in enumerate(file_iter, start=1):
            for obj in read_jsonl(file_path):
                if not isinstance(obj, dict):
                    continue
                translations = obj.get("translations")
                if not isinstance(translations, dict):
                    continue
                ind = obj.get("answer")
                bal = translations.get("balinese")
                cbr = translations.get("cirebonese")
                doc_id = obj.get("id") or obj.get("custom_id")
                if not isinstance(doc_id, str):
                    continue
                if (
                    not isinstance(ind, str)
                    or not isinstance(bal, str)
                    or not isinstance(cbr, str)
                ):
                    continue
                ind = ind.strip()
                bal = bal.strip()
                cbr = cbr.strip()
                if not ind or not bal or not cbr:
                    continue

                doc_id = doc_id.strip()
                if not doc_id:
                    continue
                if balinese_ids is not None and doc_id not in balinese_ids:
                    continue
                if cirebonese_ids is not None and doc_id not in cirebonese_ids:
                    continue
                if doc_id in seen:
                    continue

                seen.add(doc_id)
                yield {
                    "id": doc_id,
                    "indonesian": ind,
                    "balinese": bal,
                    "cirebonese": cbr,
                }

            if show_progress and idx % 10 == 0:
                print(
                    f"[build_parallel_hf_dataset] Processed {idx}/{total_files} files, "
                    f"current unique records: {len(seen)}",
                    flush=True,
                )

        if show_progress:
            print(
                f"[build_parallel_hf_dataset] Finished scanning {total_files} files; "
                f"total unique records: {len(seen)}",
                flush=True,
            )

    return generator


def write_hf_dataset(
    output_dir: Path, generator_fn: Callable[[], Iterable[Dict[str, str]]]
) -> int:
    ensure_directory(output_dir)
    features = Features(
        {
            "id": Value("string"),
            "indonesian": Value("string"),
            "balinese": Value("string"),
            "cirebonese": Value("string"),
        }
    )
    dataset = cast(Dataset, Dataset.from_generator(generator_fn, features=features))
    num_records = len(dataset)
    if num_records == 0:
        print(
            f"[build_parallel_hf_dataset] No records to save for {output_dir}; skipping save",
            flush=True,
        )
        return 0
    dataset.save_to_disk(str(output_dir))
    return num_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HF-style parallel dataset from topic_translations JSONLs"
    )
    parser.add_argument("--topics-dir", type=Path, default=Path("synthetic_data/raw/translations"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--balinese-clean",
        type=Path,
        default=Path("metrics/cache/balinese/translations_clean.jsonl"),
        help="Path to cleaned Balinese translations JSONL (ids used for filtering)",
    )
    parser.add_argument(
        "--cirebonese-clean",
        type=Path,
        default=Path("metrics/cache/cirebonese/translations_clean.jsonl"),
        help="Path to cleaned Cirebonese translations JSONL (ids used for filtering)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    args = parser.parse_args()

    bal_ids = load_clean_ids(args.balinese_clean, "Balinese")
    cbr_ids = load_clean_ids(args.cirebonese_clean, "Cirebonese")

    generator_fn = iter_parallel_records(
        args.topics_dir,
        balinese_ids=bal_ids,
        cirebonese_ids=cbr_ids,
        show_progress=not args.no_progress,
    )
    num_records = write_hf_dataset(args.output_dir, generator_fn)
    print({"num_records": num_records})


if __name__ == "__main__":
    main()
