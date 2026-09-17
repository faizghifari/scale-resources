#!/usr/bin/env python
"""OCR + structured parsing of the scanned Aceh dictionaries with the local VLM.

Uses the multimodal Qwen served on the Hermes llama.cpp endpoint directly (the
Hermes agent's own vision tool is wired to an external API). Two stages, both
resumable -- a page whose output file exists is skipped:

  ocr    PDF page -> 200 dpi PNG -> faithful transcription   .../vlm_ocr/<vol>/page_NNNN.txt
  parse  transcription -> JSON entries                        .../vlm_parse/<vol>/page_NNNN.json

Why re-OCR at all: archive.org's OCR of these books drops almost every Acehnese
diacritic (pupôk -> pupok, lôn -> Ion) and the homograph numbers; the VLM keeps
both (pilot, Jilid 2 page 200).

    python -m scaleres.dataprep.ocr_dictionary ocr   --vols jilid1 jilid2 ind_aceh --workers 4
    python -m scaleres.dataprep.ocr_dictionary parse --vols jilid1 jilid2 ind_aceh --workers 4
    python -m scaleres.dataprep.ocr_dictionary status
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "dataset/raw/dict/_source/archive_org"
RESULTS = ROOT / "autoresearch/experiments/results"
URL = "http://143.248.188.121:10080/v1/chat/completions"

VOLS = {
    "jilid1": {"pdf": "aceh_ind_jilid1.pdf", "dir": "aceh_ind", "title": "Kamus Bahasa Aceh-Indonesia, Jilid 1"},
    "jilid2": {"pdf": "aceh_ind_jilid2.pdf", "dir": "aceh_ind", "title": "Kamus Bahasa Aceh-Indonesia, Jilid 2"},
    "ind_aceh": {"pdf": "ind_aceh.pdf", "dir": "ind_aceh", "title": "Kamus Indonesia-Aceh"},
}

OCR_PROMPT = (
    "This is a page from the printed dictionary \"{title}\". "
    "Transcribe ALL text on the page exactly as printed, in reading order (if there are two columns, "
    "finish the left column before the right). Preserve every diacritic exactly (è é ë ô ö ê œ etc.), "
    "punctuation, dashes, arrows (write -> for an arrow), numbers and abbreviations. "
    "When a word is split across a line break with a hyphen, join the syllables into one word; but when "
    "both halves are whole words (reduplication such as daun-daunan, ubun-ubun, bit-bit), keep the hyphen. "
    "Output plain text only, one dictionary entry per paragraph separated by a blank line. "
    "Do not translate, correct, or add anything. If the page has no text, output EMPTY."
)

PARSE_ACE_IND = """Below is OCR text of one page of the Acehnese-Indonesian dictionary "{title}" (Balai Pustaka / Pusat Bahasa). Convert every dictionary entry into JSON. Rules:
- One object per headword sense block. Fields: "headword" (Acehnese, keep diacritics exactly), "homograph" (leading number like 1/2, else null), "pos" (abbreviation as printed: n, v, a, adv, kep, pron, num, p, ...; null if none), "gloss_ind" (the Indonesian definition only, before the colon that introduces examples), "examples" (list of {{"ace": ..., "ind": ...}}; an Acehnese example is followed by its Indonesian translation; "-" stands for the headword: expand it, and "- -" is the reduplicated headword joined by a hyphen, e.g. pura-pura), "derived" (list of {{"form": ..., "gloss_ind": ...}} for sub-entries like "mu-, berpupur", keep the form as printed), "see_also" (words after "->").
- A cross-reference-only entry like "pupu -> pupe" gives {{"headword":"pupu","see_also":["pupe"]}} with other fields null/empty.
- Text at the top of the page that continues an entry from the previous page: output {{"continuation": "<text>"}}.
- Running page header words and page numbers are NOT entries. Front matter (preface, abbreviation lists, etc.) is not entries: for such a page return [{{"non_entry_page": true}}].
- Copy text exactly; do not correct spelling, do not invent translations. If unsure how to split something, keep it inside gloss_ind.
Return ONLY a JSON array.

OCR text:
"""

PARSE_IND_ACE = """Below is OCR text of one page of the Indonesian-Acehnese dictionary "{title}". Convert every dictionary entry into JSON. Rules:
- One object per Indonesian headword or sub-headword. Fields: "headword_ind" (Indonesian, as printed), "pos" (abbreviation as printed, else null), "ace" (list of Acehnese equivalents, each exactly as printed with all diacritics; split alternatives separated by ";" or ","), "examples" (list of {{"ind": ..., "ace": ...}} if phrase examples are given; "---" or "-" stands for the headword: expand it), "note" (any other text, else null).
- Text at the top that continues an entry from the previous page: {{"continuation": "<text>"}}.
- Page headers/page numbers are not entries. Front matter pages: return [{{"non_entry_page": true}}].
- Copy text exactly; do not correct spelling or invent equivalents.
Return ONLY a JSON array.

OCR text:
"""


def pages(vol: str) -> int:
    out = subprocess.run(["pdfinfo", str(BASE / VOLS[vol]["pdf"])], capture_output=True, text=True).stdout
    return int(next(l.split()[-1] for l in out.splitlines() if l.startswith("Pages:")))


def looped(text: str) -> bool:
    """A degenerate decode: some non-trivial line repeated 4+ times (a real page never does)."""
    from collections import Counter
    c = Counter(l for l in text.split("\n") if len(l.strip()) > 3)
    return bool(c) and c.most_common(1)[0][1] >= 4


def call(messages, max_tokens, penalty=None):
    for attempt in range(4):
        try:
            body = {"model": "x", "temperature": 0, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
            if penalty:
                body.update({"repeat_penalty": penalty, "repeat_last_n": 256})
            r = requests.post(URL, json=body, timeout=1200)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"], j.get("usage", {}), j["choices"][0].get("finish_reason")
        except Exception as e:  # server busy / transient
            err = e
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {err}")


def ocr_page(vol: str, p: int) -> str:
    d = BASE / "vlm_ocr" / vol
    out = d / f"page_{p:04d}.txt"
    if out.exists():
        return "skip"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", "200", "-png", "-singlefile",
                        str(BASE / VOLS[vol]["pdf"]), f"{td}/pg"], check=True)
        img = base64.b64encode(Path(f"{td}/pg.png").read_bytes()).decode()
    t = time.time()
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        {"type": "text", "text": OCR_PROMPT.format(title=VOLS[vol]["title"])}]}]
    # p99 of a real page is ~810 completion tokens; 2000 only ever truncates a loop
    text, usage, fin = call(msgs, 2000)
    retried = False
    if fin == "length" or looped(text):
        retried = True
        text, usage, fin = call(msgs, 2000, penalty=1.1)
    d.mkdir(parents=True, exist_ok=True)
    flagged = fin == "length" or looped(text)
    (d / f"page_{p:04d}.meta.json").write_text(json.dumps(
        {"secs": round(time.time() - t, 1), "usage": usage, "finish_reason": fin,
         "retried_with_penalty": retried, "flagged_loop": flagged}))
    out.write_text(text)
    return ("FLAGGED_LOOP " if flagged else "") + (fin or "ok") + (" (retried)" if retried else "")


def repair_json(body: str) -> str:
    """Insert the comma the model forgets between two string members.

    jilid1 p365 failed twice, not from truncation -- the array closed properly -- but from
    output like `"ace": "boh-,"` followed on the next line by `"ind": "..."`. In valid JSON
    a closing quote followed by a newline and another quote *always* needs a comma between
    them, whether the members are object fields or array elements, so this cannot turn a
    valid document into a different valid one: anything it rewrites was already malformed.
    """
    return re.sub(r'"(\s*\n\s*)"', r'",\1"', body)


def parse_page(vol: str, p: int) -> str:
    src = BASE / "vlm_ocr" / vol / f"page_{p:04d}.txt"
    d = BASE / "vlm_parse" / vol
    out = d / f"page_{p:04d}.json"
    if out.exists() or not src.exists():
        return "skip"
    text = src.read_text()
    d.mkdir(parents=True, exist_ok=True)
    if text.strip() in ("", "EMPTY"):
        out.write_text("[]")
        return "empty"
    tmpl = PARSE_IND_ACE if VOLS[vol]["dir"] == "ind_aceh" else PARSE_ACE_IND
    raw, usage, fin = call([{"role": "user", "content": tmpl.format(title=VOLS[vol]["title"]) + text}], 5000)
    if fin == "length":
        raw, usage, fin = call([{"role": "user", "content": tmpl.format(title=VOLS[vol]["title"]) + text}],
                               5000, penalty=1.1)
    body = raw[raw.find("["):raw.rfind("]") + 1]
    try:
        data = json.loads(body)
    except Exception:
        try:
            data = json.loads(repair_json(body))
        except Exception:
            (d / f"page_{p:04d}.bad.txt").write_text(raw)
            return "bad_json"
    out.write_text(json.dumps(data, ensure_ascii=False))
    return fin or "ok"


def run(stage: str, vols: list[str], workers: int):
    fn = ocr_page if stage == "ocr" else parse_page
    jobs = [(v, p) for v in vols for p in range(1, pages(v) + 1)]
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, v, p): (v, p) for v, p in jobs}
        for f in as_completed(futs):
            v, p = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = f"ERROR {e}"
            done += 1
            if res != "skip" or done % 200 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {stage} {v} p{p:04d} {res}  "
                      f"({done}/{len(jobs)}, {time.time()-t0:.0f}s)", flush=True)


ARCHIVE_TXT = {"jilid1": "Kamus_Aceh_Indonesia_Jilid1_OCR.txt",
               "jilid2": "Kamus_Aceh_Indonesia_Jilid2_OCR.txt",
               "ind_aceh": "Kamus_Indonesia_Aceh_OCR.txt"}
# page_NNNN.prerepair.txt backups sit beside the pages; a bare page_* glob counts them
PAGE_FILE = re.compile(r"page_\d+\.(txt|json)")

QC_MIN_WORDS = 40          # shorter pages are front matter / plates, no signal
QC_MIN_COVERAGE = 0.55     # word-level hit rate against archive.org's own OCR; good pages 0.85-0.98
QC_GOOD_ENOUGH = 0.80      # a repaired page scoring this is normal; stop trying variants
QC_LEX_DROP = 0.20         # how far below the volume median in-lexicon rate is suspicious

# 4-gram coverage was the first QC signal and is kept only for reference in the report: on
# pages that are lists of synonyms or place names (most of ind_aceh) a single differing word
# destroys most 4-grams, so it flagged 625/1176 pages that word-level scoring shows are clean.
WORD3 = re.compile(r"[a-z']{3,}")
WORD4 = re.compile(r"[a-z']{4,}")


def _de(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9' ]+", " ", re.sub(r"[̀-ͯ]", "", s).lower())


def archive_words(vol: str) -> set:
    """Vocabulary of archive.org's own OCR of the volume, diacritics stripped."""
    return set(WORD3.findall(_de((BASE / ARCHIVE_TXT[vol]).read_text(encoding="utf-8", errors="ignore"))))


def volume_lexicon(vol: str, min_count: int = 3) -> set:
    """Words the VLM read on >=3 pages of this volume -- its own consistent vocabulary.

    A garbled or mirrored page reads mostly strings no other page contains, so it scores
    far below the volume median even where archive.org's OCR is too poor to compare against.
    """
    from collections import Counter
    c = Counter()
    for f in sorted((BASE / "vlm_ocr" / vol).glob("page_*.txt")):
        c.update(set(WORD4.findall(_de(f.read_text(errors="ignore")))))
    return {w for w, n in c.items() if n >= min_count}


LOOP_MAX_UNIT = 12         # the repeating unit of a loop can be a whole clause
LOOP_MIN = 3               # a unit repeated this many times in a row is a loop


def loop_span(toks: list, max_unit: int = LOOP_MAX_UNIT, min_reps: int = LOOP_MIN):
    """Longest back-to-back repetition of a 1..max_unit token unit, as (start, end) or None.

    The unit has to be allowed to be several tokens long. jilid2 p520 looped
    "- 1 yuta = 13 sampai 15" -- seven tokens -- about 120 times, and both a 4-token
    window and a single-token equality test miss it completely: no individual token is
    ever adjacent to a copy of itself. It scored 0.962 and was accepted.
    """
    n, best = len(toks), None
    for u in range(1, max_unit + 1):
        i = 0
        while i + u * min_reps <= n:
            j = i + u
            while j + u <= n and toks[j:j + u] == toks[i:i + u]:
                j += u
            if (j - i) // u >= min_reps:
                if best is None or (j - i) > best[1] - best[0]:
                    best = (i, j)
                i = j
            else:
                i += 1
    return best


def repetition(text: str) -> float:
    """How much of a page is taken up by its most repetitive element.

    Counted over tokens as well as lines, because a loop can fill a single long line:
    jilid1 p211 came back as five lines, one of which was "hub" repeated 1,900 times,
    which a line-level count reads as one line in five (0.2, under the reject threshold)
    while coverage reads it as ~1.0 because "hub" is a real word in the volume.
    """
    from collections import Counter
    lines = [" ".join(l.split()) for l in text.split("\n") if len(l.strip()) > 3]
    rep = Counter(lines).most_common(1)[0][1] / len(lines) if lines else 0.0
    toks = [t.lower() for t in text.split()]
    if len(toks) >= 20:
        span = loop_span(toks)
        if span:
            rep = max(rep, (span[1] - span[0]) / len(toks))
    return round(rep, 3)


def page_quality(vol: str, p: int, ref: set, lex: set | None = None, text: str | None = None):
    """(archive word coverage, in-lexicon rate, words, top_repeat_ratio)."""
    from collections import Counter
    f = BASE / "vlm_ocr" / vol / f"page_{p:04d}.txt"
    if text is None:
        if not f.exists():
            return None
        text = f.read_text()
    d = _de(text)
    w3, w4 = WORD3.findall(d), WORD4.findall(d)
    cov = sum(w in ref for w in w3) / len(w3) if w3 else 0.0
    fwd = (sum(w in lex for w in w4) / len(w4) if w4 else 0.0) if lex is not None else None
    return round(cov, 3), (round(fwd, 3) if fwd is not None else None), len(w3), repetition(text)


def qc(vols):
    out = {}
    for vol in vols:
        ref, lex = archive_words(vol), volume_lexicon(vol)
        rows = {}
        for p in range(1, pages(vol) + 1):
            q = page_quality(vol, p, ref, lex)
            if q is None:
                continue
            cov, fwd, nw, rep = q
            rows[p] = {"coverage": cov, "in_lexicon": fwd, "words": nw, "top_repeat_ratio": rep}
        scored = [r for r in rows.values() if r["words"] >= QC_MIN_WORDS]
        med = sorted(r["in_lexicon"] for r in scored)[len(scored) // 2] if scored else 0
        suspects = [p for p, r in sorted(rows.items())
                    if r["words"] >= QC_MIN_WORDS
                    and (r["coverage"] < QC_MIN_COVERAGE or r["top_repeat_ratio"] > 0.6
                         or (r["in_lexicon"] < med - QC_LEX_DROP and r["coverage"] < 0.75))]
        blank = [p for p, r in sorted(rows.items()) if r["words"] < 5]
        out[vol] = {"pages_scored": len(rows), "median_in_lexicon": med, "suspects": suspects,
                    "blank_pages": blank, "pages": rows}
        cov = sorted(r["coverage"] for r in scored)
        print(f"{vol}: {len(rows)} pages scored, median word coverage {cov[len(cov) // 2] if cov else 0}, "
              f"median in-lexicon {med}, {len(blank)} blank, {len(suspects)} suspects: {suspects}")
    (RESULTS / "S2_14_ocr_qc.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {RESULTS / 'S2_14_ocr_qc.json'}")


def sharpen(im):
    """Rescue a faint page: some scans caught mirrored show-through from the reverse side,
    leaving the real text as a pale double exposure the model reads as noise."""
    from PIL import ImageEnhance, ImageFilter, ImageOps
    g = ImageOps.autocontrast(im.convert("L"), cutoff=(1, 12))
    return ImageEnhance.Contrast(g).enhance(2.2).filter(ImageFilter.UnsharpMask(2, 180, 3))


VARIANTS = {"flop": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
            "rot180": lambda im: im.transpose(Image.ROTATE_180),
            "flop_rot180": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180),
            "flip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
            "sharpen": sharpen,
            "flop_sharpen": lambda im: sharpen(im.transpose(Image.FLIP_LEFT_RIGHT))}

def truncate_loop(text: str) -> str:
    """Cut a page at the point the model started repeating itself.

    A degraded page often reads correctly for a few entries and then loops to the token
    limit ("duya n saudara ... -ad, -ad, -ad"). The readable prefix is worth keeping, so
    the loop is cut off rather than the whole page thrown away. The loop is found with the
    same `loop_span` the score uses, so a page cannot be truncated by one definition of a
    loop and then rewarded under another.
    """
    out, run, prev = [], 0, None
    for line in text.split("\n"):
        key = " ".join(line.split()).lower()
        run = run + 1 if key and key == prev else 0
        if run >= LOOP_MIN - 1:
            break
        prev = key
        toks = line.split()
        span = loop_span([t.lower() for t in toks])
        if span:
            out.append(" ".join(toks[:span[0]]).strip())
            return "\n".join(out).strip()
        out.append(line)
    return "\n".join(out).strip()


def volume_median_words(vol: str) -> int:
    n = sorted(len(WORD3.findall(_de(f.read_text(errors="ignore"))))
               for f in (BASE / "vlm_ocr" / vol).glob("page_*.txt"))
    return max(n[len(n) // 2], 1) if n else 1


def repair_score(q, median_words: int) -> float:
    """Correct words read, as a fraction of a normal page for this volume.

    Coverage alone is the wrong objective and accepted two bad repairs: a page that loops
    one phrase 285 times scores 1.0 because every word is in the reference, and so does a
    page where the model gave up after two words. Rewarding completeness as well as accuracy,
    and rejecting repetition outright, fixes both.

    Completeness must not be allowed to outvote accuracy, though. On jilid2 p520 a 666-word
    read that had degenerated into English ("enduce for love; force; seduce") scored 0.428
    and beat an 83-word read that was perfectly correct (0.18), because length made up for
    a coverage of 0.43. Accuracy is therefore a gate, not a factor: below QC_MIN_COVERAGE
    the page is unreadable in this orientation and its length is not evidence of anything.
    """
    cov, _fwd, nw, rep = q
    if rep > 0.25 or nw < 20 or cov < QC_MIN_COVERAGE:
        return 0.0
    return round(cov * min(nw / median_words, 1.0), 3)


def repair_page(vol: str, p: int, ref: set, lex: set | None = None, median_words: int = 1):
    """Re-OCR a suspect page in each orientation; keep the best-scoring version.

    The original orientation is re-read too, so a page whose stored text was already
    replaced by a worse repair can recover, and the pre-repair text is kept beside it.
    """
    import tempfile as tf
    d = BASE / "vlm_ocr" / vol
    txt = d / f"page_{p:04d}.txt"
    before = page_quality(vol, p, ref, lex)
    base_score = repair_score(before, median_words) if before else 0.0
    best = (base_score, "stored", None)
    with tf.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", "200", "-png", "-singlefile",
                        str(BASE / VOLS[vol]["pdf"]), f"{td}/pg"], check=True)
        for name, fn in [("original", lambda im: im)] + list(VARIANTS.items()):
            if best[0] >= QC_GOOD_ENOUGH:      # mirrored is by far the common defect; stop early
                break
            fn(Image.open(f"{td}/pg.png")).save(f"{td}/{name}.png")
            img = base64.b64encode(Path(f"{td}/{name}.png").read_bytes()).decode()
            text, usage, fin = call([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                {"type": "text", "text": OCR_PROMPT.format(title=VOLS[vol]["title"])}]}], 2000)
            text = truncate_loop(text)
            s = repair_score(page_quality(vol, p, ref, lex, text=text), median_words)
            if s > best[0]:
                best = (s, name, text)
    if best[1] == "stored" and txt.exists():
        # No re-read beat what is on disk. If what is on disk is a loop, keep only the
        # readable prefix: a few real entries are worth more to the parser than a page
        # of "hub hub hub", which would otherwise become garbage dictionary rows.
        stored = txt.read_text()
        if repetition(stored) > 0.25:
            cut = truncate_loop(stored)
            if len(cut.split()) >= 5 and len(cut) < len(stored):
                best = (repair_score(page_quality(vol, p, ref, lex, text=cut), median_words),
                        "truncated", cut)
    if best[2] is not None:
        # Back up the best superseded version, not merely the first one ever seen. The
        # backup used to be written only if absent, so each run that scored a worse text
        # higher ratcheted the page down and the good version was unrecoverable: p520's
        # correct 1,071-word read was lost exactly that way.
        bk = d / f"page_{p:04d}.prerepair.txt"
        if txt.exists():
            old = txt.read_text()
            if not bk.exists() or repair_score(
                    page_quality(vol, p, ref, lex, text=old), median_words) > repair_score(
                    page_quality(vol, p, ref, lex, text=bk.read_text()), median_words):
                bk.write_text(old)
        txt.write_text(best[2])
        mf = d / f"page_{p:04d}.meta.json"
        m = json.loads(mf.read_text()) if mf.exists() else {}
        m.update({"repaired_variant": best[1], "repaired_score": best[0],
                  "score_before": base_score})
        mf.write_text(json.dumps(m))
    return best[0], best[1], base_score


def repair(vols, workers, only=None):
    rep = json.loads((RESULTS / "S2_14_ocr_qc.json").read_text())
    for vol in vols:
        sus = only if only else rep.get(vol, {}).get("suspects", [])
        sus = [p for p in sus if (BASE / "vlm_ocr" / vol / f"page_{p:04d}.txt").exists()]
        if not sus:
            print(f"{vol}: no suspects"); continue
        ref, lex = archive_words(vol), volume_lexicon(vol)
        mw = volume_median_words(vol)
        fixed = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(repair_page, vol, p, ref, lex, mw): p for p in sus}
            for f in as_completed(futs):
                p = futs[f]
                try:
                    cov, variant, before = f.result()
                except Exception as e:
                    print(f"  {vol} p{p:04d} ERROR {e}"); continue
                fixed += variant != "stored"
                print(f"  {vol} p{p:04d} score {before} -> {cov} via {variant}", flush=True)
        print(f"{vol}: {fixed}/{len(sus)} pages re-read better than the stored text")


def status():
    for v in VOLS:
        n = pages(v)
        o = sum(1 for f in (BASE / "vlm_ocr" / v).glob("page_*.txt")
                if PAGE_FILE.fullmatch(f.name)) if (BASE / "vlm_ocr" / v).exists() else 0
        pr = sum(1 for f in (BASE / "vlm_parse" / v).glob("page_*.json")
                 if PAGE_FILE.fullmatch(f.name)) if (BASE / "vlm_parse" / v).exists() else 0
        bad = len(list((BASE / "vlm_parse" / v).glob("*.bad.txt"))) if (BASE / "vlm_parse" / v).exists() else 0
        print(f"{v:9s} pages {n:5d}  ocr {o:5d}  parsed {pr:5d}  bad_json {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ocr", "parse", "status", "qc", "repair"])
    ap.add_argument("--vols", nargs="+", default=list(VOLS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pages", nargs="+", type=int,
                    help="repair only these page numbers (needs a single --vols)")
    a = ap.parse_args()
    if a.stage == "status":
        status()
    elif a.stage == "qc":
        qc(a.vols)
    elif a.stage == "repair":
        if a.pages and len(a.vols) != 1:
            sys.exit("--pages needs exactly one --vols")
        repair(a.vols, a.workers, a.pages)
    else:
        run(a.stage, a.vols, a.workers)


if __name__ == "__main__":
    sys.exit(main())
