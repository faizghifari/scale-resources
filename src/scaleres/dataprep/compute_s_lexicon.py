import argparse
import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple, cast

import matplotlib.pyplot as plt
from datasets import DatasetDict, load_from_disk

from .quality_utils import basic_tokenize, ensure_directory

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


_WORKER_STATE: Dict[str, object] = {}


def _init_worker(
    lexicon: Dict[str, List[str]],
    target_stop: Dict[str, None],
    gloss_index: Dict[str, List[str]],
    idn_stopwords: Sequence[str],
    ngram: int,
) -> None:
    _WORKER_STATE["lexicon"] = lexicon
    _WORKER_STATE["target_stop"] = target_stop
    _WORKER_STATE["gloss_index"] = gloss_index
    _WORKER_STATE["idn_stopwords"] = list(idn_stopwords)
    _WORKER_STATE["ngram"] = ngram


def _compute_doc_metrics(
    payload: Tuple[str, List[str], List[str]],
    lexicon: Dict[str, List[str]],
    target_stop: Dict[str, None],
    gloss_index: Dict[str, List[str]],
    idn_stopwords: Sequence[str],
    ngram: int,
) -> Tuple[str, float, int, int]:
    doc_id, idn_sents, tgt_sents = payload
    pair_count = min(len(idn_sents), len(tgt_sents))
    total_roots = 0
    matched_roots = 0
    for i in range(pair_count):
        _, matched, roots = compute_sentence_s_lexicon(
            idn_sents[i],
            tgt_sents[i],
            lexicon,
            target_stop,
            gloss_index,
            idn_stopwords,
            n=ngram,
        )
        matched_roots += matched
        total_roots += roots
    s_doc = matched_roots / total_roots if total_roots > 0 else 0.0
    return doc_id, s_doc, matched_roots, total_roots


def _process_doc_worker(
    payload: Tuple[str, List[str], List[str]],
) -> Tuple[str, float, int, int]:
    lexicon = cast(Dict[str, List[str]], _WORKER_STATE["lexicon"])
    target_stop = cast(Dict[str, None], _WORKER_STATE["target_stop"])
    gloss_index = cast(Dict[str, List[str]], _WORKER_STATE["gloss_index"])
    idn_stopwords = cast(Sequence[str], _WORKER_STATE["idn_stopwords"])
    ngram = cast(int, _WORKER_STATE["ngram"])
    return _compute_doc_metrics(
        payload, lexicon, target_stop, gloss_index, idn_stopwords, ngram
    )


def _progress_iter(
    iterator: Iterable[Tuple[str, float, int, int]],
    total: int,
    desc: str,
    log_interval: int,
) -> Iterator[Tuple[str, float, int, int]]:
    if tqdm is not None:
        with tqdm(
            total=total,
            desc=desc,
            mininterval=1.0,
            smoothing=0.05,
            dynamic_ncols=True,
        ) as bar:
            for item in iterator:
                bar.update(1)
                yield item
        return

    processed = 0
    start = time.time()
    for item in iterator:
        processed += 1
        if (
            processed == 1
            or processed % max(1, log_interval) == 0
            or processed == total
        ):
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            percent = (processed / total) * 100 if total else 0.0
            print(
                f"[s_lexicon] {desc}: {processed}/{total} docs ({percent:.1f}%) at {rate:.1f} docs/s",
                flush=True,
            )
        yield item


def _derived_chunksize(total_docs: int, requested: int, workers: int) -> int:
    if total_docs <= 0:
        return 1
    worker_bucket = max(1, workers)
    adaptive_cap = max(1, total_docs // (worker_bucket * 256))
    return max(1, min(requested, adaptive_cap, total_docs))


def _load_existing_results(path: Path) -> Tuple[Set[str], List[float]]:
    processed: Set[str] = set()
    scores: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj.get("doc_id")
            score = obj.get("s_lexicon_doc")
            if isinstance(doc_id, str):
                processed.add(doc_id)
            if isinstance(score, (int, float)):
                scores.append(float(score))
    return processed, scores


def load_json(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): list(v) for k, v in data.items() if isinstance(v, (list, tuple))}


def build_target_lexicon(
    from_idn: Path, to_idn: Path, idn_stopwords: Sequence[str]
) -> Tuple[Dict[str, List[str]], Dict[str, None], Dict[str, List[str]]]:
    idn_to_target = load_json(from_idn)
    target_to_idn = load_json(to_idn)

    lexicon: Dict[str, List[str]] = {}
    for d in (idn_to_target, target_to_idn):
        for key, vals in d.items():
            key_norm = key.strip().lower()
            if not key_norm:
                continue
            merged = list(lexicon.get(key_norm, []))
            for v in vals:
                v_norm = v.strip().lower()
                if not v_norm:
                    continue
                if v_norm not in merged:
                    merged.append(v_norm)
            lexicon[key_norm] = merged

    target_stop: Dict[str, None] = {}
    gloss_index: Dict[str, List[str]] = {}
    idn_stop_set = {w.lower() for w in idn_stopwords}

    for idn_word, targets in idn_to_target.items():
        if idn_word.lower() in idn_stop_set:
            for tgt in targets:
                tgt_norm = tgt.strip().lower()
                if tgt_norm:
                    target_stop[tgt_norm] = None

    for root, glosses in lexicon.items():
        for gloss in glosses:
            tokens = [t.strip().lower() for t in gloss.split() if t.strip()]
            for tok in tokens:
                roots = gloss_index.setdefault(tok, [])
                if root not in roots:
                    roots.append(root)

    return lexicon, target_stop, gloss_index


def nasalization_variants(root: str) -> List[str]:
    variants = {root}
    mapping = {"t": "n", "s": "ny", "k": "ng"}
    lower = root.lower()
    for src, dst in mapping.items():
        if lower.startswith(src):
            variants.add(dst + lower[len(src) :])
    return sorted(variants)


def ngram_multiset(s: str, n: int) -> Counter:
    s = s.lower()
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def ngram_recall(root: str, gen: str, n: int = 3) -> float:
    if not root or not gen:
        return 0.0
    best = 0.0
    for variant in nasalization_variants(root):
        r_ngrams = ngram_multiset(variant, n)
        if not r_ngrams:
            continue
        g_ngrams = ngram_multiset(gen, n)
        if not g_ngrams:
            continue
        inter = 0
        for g in g_ngrams:
            if g in r_ngrams:
                inter += min(r_ngrams[g], g_ngrams[g])
        denom = sum(r_ngrams.values())
        if denom == 0:
            continue
        score = inter / denom
        if score > best:
            best = score
    return best


def best_match_score(root: str, target_tokens: Sequence[str], n: int = 3) -> float:
    best = 0.0
    for tok in target_tokens:
        s = ngram_recall(root, tok, n=n)
        if s > best:
            best = s
    return best


def compute_sentence_s_lexicon(
    idn_sentence: str,
    target_sentence: str,
    lexicon: Dict[str, List[str]],
    target_stop: Dict[str, None],
    gloss_index: Dict[str, List[str]],
    idn_stopwords: Sequence[str],
    n: int = 3,
) -> Tuple[float, int, int]:
    idn_tokens = [t.lower() for t in basic_tokenize(idn_sentence)]
    target_tokens = [t.lower() for t in basic_tokenize(target_sentence)]

    idn_stop_set = {w.lower() for w in idn_stopwords}

    candidate_roots: List[str] = []
    for tok in idn_tokens:
        if tok in idn_stop_set:
            continue
        roots = gloss_index.get(tok)
        if roots:
            candidate_roots.extend(roots)

    unique_roots: List[str] = []
    seen = set()
    for r in candidate_roots:
        if r in seen or r in target_stop:
            continue
        seen.add(r)
        unique_roots.append(r)

    if not unique_roots:
        return 0.0, 0, 0

    matched = 0
    for root in unique_roots:
        score = best_match_score(root, target_tokens, n=n)
        if score > 0.5:
            matched += 1

    s = matched / len(unique_roots)
    return s, matched, len(unique_roots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute S_lexicon doc-level scores")
    parser.add_argument("--lang", choices=["balinese", "cirebonese"], required=True)
    parser.add_argument(
        "--hf",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "HF-style parallel dataset(s) with columns: id, indonesian, balinese, cirebonese. "
            "Each path can be a directory containing data.jsonl or a JSONL file itself."
        ),
    )
    parser.add_argument(
        "--output-doc",
        type=Path,
        required=True,
        help="Output JSONL with doc-level S_lexicon scores",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        required=True,
        help="Path to save distribution plot (e.g., png)",
    )
    parser.add_argument("--idn-bali", type=Path, default=Path("dict/idn_bali.json"))
    parser.add_argument("--bali-idn", type=Path, default=Path("dict/bali_idn.json"))
    parser.add_argument("--idn-cbn", type=Path, default=Path("dict/idn_cbn.json"))
    parser.add_argument("--cbn-idn", type=Path, default=Path("dict/cbn_idn.json"))
    parser.add_argument(
        "--idn-stopwords",
        type=Path,
        default=None,
        help=(
            "Optional Indonesian stopword list (one per line). If not provided, use a small built-in list."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional quality threshold for S_lexicon_doc; sets quality_decision field.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on number of documents to process.",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=3,
        help="Character n-gram size for soft matching (default 3).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel workers for doc scoring (default: available CPU cores).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=16,
        help=(
            "Number of docs per worker chunk when multiprocessing is enabled "
            "(smaller = more responsive progress)."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5000,
        help="Rows/docs between manual progress logs when tqdm is unavailable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip doc_ids already present in the output JSONL and append new scores.",
    )

    args = parser.parse_args()

    if args.idn_stopwords is not None:
        with args.idn_stopwords.open("r", encoding="utf-8") as f:
            idn_stopwords = [ln.strip() for ln in f if ln.strip()]
    else:
        idn_stopwords = [
            "yang",
            "dan",
            "di",
            "ke",
            "dari",
            "untuk",
            "dengan",
            "atau",
            "pada",
            "ini",
            "itu",
            "sebagai",
            "juga",
        ]

    if args.lang == "balinese":
        lexicon, target_stop, gloss_index = build_target_lexicon(
            args.idn_bali, args.bali_idn, idn_stopwords
        )
        target_column = "balinese"
    else:
        lexicon, target_stop, gloss_index = build_target_lexicon(
            args.idn_cbn, args.cbn_idn, idn_stopwords
        )
        target_column = "cirebonese"

    processed_doc_ids: Set[str] = set()
    existing_scores: List[float] = []
    if args.resume:
        if args.output_doc.exists():
            processed_doc_ids, existing_scores = _load_existing_results(args.output_doc)
            print(
                f"[s_lexicon] Resume enabled: skipping {len(processed_doc_ids)} previously scored docs",
                flush=True,
            )
        else:
            print(
                "[s_lexicon] Resume requested but no prior output found; starting fresh",
                flush=True,
            )
    elif args.output_doc.exists():
        print(
            "[s_lexicon] Warning: output file exists and will be overwritten (pass --resume to append)",
            flush=True,
        )

    idn_by_doc: Dict[str, List[str]] = defaultdict(list)
    tgt_by_doc: Dict[str, List[str]] = defaultdict(list)

    total_rows = 0
    for path in args.hf:
        print(f"[s_lexicon] Loading dataset from {path}", flush=True)
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            split_items = ds.items()
        else:
            split_items = [("data", ds)]

        for split_name, split in split_items:
            split_len = len(split)
            print(
                f"[s_lexicon] Scanning split '{split_name}' with {split_len} rows",
                flush=True,
            )
            for idx in range(split_len):
                row = split[idx]
                if not isinstance(row, dict):
                    continue
                doc_id = row.get("id")
                ind_text = row.get("indonesian")
                tgt_text = row.get(target_column)
                if (
                    not isinstance(doc_id, str)
                    or not isinstance(ind_text, str)
                    or not isinstance(tgt_text, str)
                ):
                    continue
                idn_by_doc[doc_id].append(ind_text)
                tgt_by_doc[doc_id].append(tgt_text)
                total_rows += 1
                if total_rows % max(1, args.log_interval) == 0:
                    print(
                        f"[s_lexicon] Loaded {total_rows} sentence pairs so far",
                        flush=True,
                    )

    doc_ids = sorted(set(idn_by_doc.keys()) & set(tgt_by_doc.keys()))
    if processed_doc_ids:
        doc_ids = [doc_id for doc_id in doc_ids if doc_id not in processed_doc_ids]
    if args.max_docs is not None and args.max_docs > 0:
        doc_ids = doc_ids[: args.max_docs]

    doc_payloads = [
        (doc_id, idn_by_doc[doc_id], tgt_by_doc[doc_id]) for doc_id in doc_ids
    ]
    total_docs = len(doc_payloads)
    print(
        f"[s_lexicon] Prepared {total_docs} new docs for scoring (already done: {len(processed_doc_ids)})",
        flush=True,
    )

    scores: List[float] = list(existing_scores)

    if total_docs == 0:
        if scores:
            print(
                "[s_lexicon] No new docs to score; regenerating plot from existing results",
                flush=True,
            )
        else:
            print("[s_lexicon] No docs available for scoring", flush=True)
    else:
        num_workers = max(1, min(args.num_workers, total_docs or 1))
        requested_chunksize = max(1, args.chunksize)
        effective_chunksize = _derived_chunksize(
            total_docs, requested_chunksize, num_workers
        )
        print(
            f"[s_lexicon] Using {num_workers} worker(s) with chunksize {effective_chunksize}",
            flush=True,
        )

        def iter_scores() -> Iterator[Tuple[str, float, int, int]]:
            if num_workers == 1:
                for payload in doc_payloads:
                    yield _compute_doc_metrics(
                        payload,
                        lexicon,
                        target_stop,
                        gloss_index,
                        idn_stopwords,
                        args.ngram,
                    )
                return

            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(lexicon, target_stop, gloss_index, idn_stopwords, args.ngram),
            ) as pool:
                for result in pool.imap(
                    _process_doc_worker, doc_payloads, chunksize=effective_chunksize
                ):
                    yield result

        ensure_directory(args.output_doc.parent)
        existing_file = args.output_doc.exists()
        file_mode = "a" if args.resume and existing_file else "w"
        with args.output_doc.open(file_mode, encoding="utf-8") as out:
            score_iter = iter_scores()
            for doc_id, s_doc, matched_roots, total_roots in _progress_iter(
                score_iter,
                total_docs,
                "Scoring docs",
                args.log_interval,
            ):
                scores.append(s_doc)

                decision = None
                if args.threshold is not None:
                    decision = "pass" if s_doc >= args.threshold else "fail"

                record = {
                    "doc_id": doc_id,
                    "s_lexicon_doc": float(s_doc),
                    "matched_roots": int(matched_roots),
                    "total_roots": int(total_roots),
                    "quality_decision": decision,
                }
                out.write(json.dumps(record) + "\n")

    if scores:
        ensure_directory(args.plot.parent)
        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=40, range=(0.0, 1.0), edgecolor="black", alpha=0.7)
        plt.xlabel("S_lexicon_doc")
        plt.ylabel("Count")
        plt.title("Distribution of S_lexicon_doc")
        plt.tight_layout()
        plt.savefig(args.plot)
        plt.close()


if __name__ == "__main__":
    main()
