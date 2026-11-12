"""Uniform random sample of Amazon Reviews 2023 - Grocery and Gourmet Food (A07, A22).

The full file (5.97 GB JSONL, 14M+ reviews) does not fit on this disk, and streaming it at the available
bandwidth took 10+ hours. Instead, reviews are drawn at uniformly random positions of the WHOLE file with
HTTP Range requests, so only the sampled reviews are ever downloaded:

1. Draw a random byte offset o in [0, file size).
2. Fetch a small window around o; the line containing o is the "previous" line, the line after it is the candidate.
3. A candidate is reached with probability proportional to the length of the previous line. Since the file is
   sorted by user, that length is correlated with the candidate (verbose users). Rejection sampling removes the
   bias exactly: accept with probability C / len(previous line), C <= the shortest possible record.
4. Duplicates (same line drawn twice) are skipped -> a uniform sample without replacement.

Offsets and acceptance draws come from one seeded generator and are processed in fixed batches,
so the sample is reproducible whatever the thread timing.

    python -m src.nlp.sample_reviews
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from src.config import ROOT

URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
       "raw/review_categories/Grocery_and_Gourmet_Food.jsonl")
SAMPLE_SIZE = 30_000
SEED = 42
C = 200          # bytes; every record has ~230+ bytes of fixed fields, so C / len(prev) <= 1 (violations are counted)
WINDOW = 4096    # bytes fetched each side of the offset; widened automatically for very long reviews
BATCH = 2_000
THREADS = 16
OUT = ROOT / "data" / "sampled" / "amazon_grocery_reviews_sample.parquet"
META = ROOT / "data" / "sampled" / "amazon_grocery_reviews_sample_meta.json"
KEEP = ["rating", "title", "text", "parent_asin", "user_id", "timestamp", "verified_purchase", "helpful_vote"]

_local = threading.local()


class Source:
    def __init__(self, url: str):
        self.url = url
        self.resolve()

    def resolve(self):
        r = requests.get(self.url, headers={"Range": "bytes=0-0"}, allow_redirects=False, timeout=30)
        self.location = r.headers.get("Location", self.url)   # signed CDN URL, refreshed if it expires
        head = requests.get(self.location, headers={"Range": "bytes=0-0"}, timeout=30)
        self.size = int(head.headers["Content-Range"].split("/")[1])

    def fetch(self, start: int, end: int) -> bytes:
        if not hasattr(_local, "session"):
            _local.session = requests.Session()
        s = _local.session
        for attempt in range(8):
            try:
                r = s.get(self.location, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
                if r.status_code == 206:
                    return r.content
                if r.status_code in (401, 403):
                    self.resolve()
            except requests.RequestException:
                pass
            time.sleep(1 + attempt)
        raise RuntimeError(f"range {start}-{end} failed")


def candidate(src: Source, o: int):
    """Return (start offset, previous-line length, candidate line) for random offset o, or None at file end."""
    w = WINDOW
    while True:
        lo, hi = max(0, o - w), min(src.size - 1, o + w)
        buf = src.fetch(lo, hi)
        rel = o - lo
        b = buf.rfind(b"\n", 0, rel)                       # newline ending the line before the one containing o
        e = buf.find(b"\n", rel)                           # newline ending the line containing o
        n = buf.find(b"\n", e + 1) if e >= 0 else -1       # newline ending the candidate
        start_ok = b >= 0 or lo == 0
        if e >= 0 and n >= 0 and start_ok:
            prev_len = e - (b + 1) + 1
            return lo + e + 1, prev_len, buf[e + 1:n]
        if hi == src.size - 1 and n < 0:                   # o lies in the last line: no next line
            return None
        w *= 4


def eligible(line: bytes):
    """A review is eligible if it parses, has a 1-5 rating and non-empty text (A07)."""
    try:
        r = json.loads(line)
    except ValueError:
        return None
    if r.get("rating") not in (1, 2, 3, 4, 5) or not str(r.get("text") or "").strip():
        return None
    return {k: r.get(k) for k in KEEP}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    src = Source(URL)
    rng = np.random.default_rng(SEED)
    kept, seen_offsets = [], set()
    stats = {"draws": 0, "rejected_by_length": 0, "duplicates": 0, "ineligible": 0, "short_prev_lines": 0,
             "prev_line_lengths": []}
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        while len(kept) < SAMPLE_SIZE:
            offsets = rng.integers(0, src.size, BATCH)
            accept_u = rng.random(BATCH)
            for res, u in zip(ex.map(lambda o: candidate(src, int(o)), offsets), accept_u):  # map keeps order
                stats["draws"] += 1
                if res is None or len(kept) >= SAMPLE_SIZE:
                    continue
                start, prev_len, line = res
                stats["prev_line_lengths"].append(prev_len)
                stats["short_prev_lines"] += prev_len < C
                if u >= C / prev_len:
                    stats["rejected_by_length"] += 1
                    continue
                if start in seen_offsets:
                    stats["duplicates"] += 1
                    continue
                r = eligible(line)
                if r is None:
                    stats["ineligible"] += 1
                    continue
                seen_offsets.add(start)
                kept.append({**r, "byte_offset": start})
            print(f"{len(kept):,} kept / {stats['draws']:,} draws, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(kept)
    df["rating"] = df.rating.astype(int)
    df.to_parquet(OUT, index=False)
    lengths = np.array(stats.pop("prev_line_lengths"))
    META.write_text(json.dumps({
        "source": URL, "file_bytes": src.size, "sample_size": len(df), "seed": SEED, "C": C, **stats,
        "acceptance_rate": round(1 - stats["rejected_by_length"] / max(len(lengths), 1), 4),
        "mean_record_bytes_est": round(float(len(lengths) / np.sum(1 / lengths)), 1),   # harmonic mean = unbiased
        "est_total_records": int(src.size / (len(lengths) / np.sum(1 / lengths))),
        "method": "random byte offsets + length-bias rejection sampling, without replacement"}, indent=2))
    print(f"kept {len(df):,} reviews -> {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
