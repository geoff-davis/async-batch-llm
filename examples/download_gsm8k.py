"""Download the GSM8K test split and write it as a gzipped JSONL file.

This is one-time setup for ``example_batch_benchmark.py``. It fetches the
canonical GSM8K test split (1,319 grade-school math word problems, each a
``{"question": ..., "answer": ...}`` record whose answer ends with a
``#### <number>`` marker) from the OpenAI ``grade-school-math`` repo and
writes it compressed to ``examples/data/gsm8k_test.jsonl.gz``.

The benchmark demo reads it back with ``aiogzip`` — the point of the demo is
async I/O on both ends, so we keep the on-disk format gzipped.

Usage::

    python examples/download_gsm8k.py          # idempotent: skips if present
    python examples/download_gsm8k.py --force  # re-download

No API keys required.
"""

import gzip
import sys
import urllib.request
from pathlib import Path

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "gsm8k_test.jsonl.gz"


def main() -> None:
    force = "--force" in sys.argv[1:]

    if OUTPUT_PATH.exists() and not force:
        size_kb = OUTPUT_PATH.stat().st_size / 1024
        print(f"Already present: {OUTPUT_PATH} ({size_kb:.0f} KiB)")
        print("Pass --force to re-download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GSM8K test split from:\n  {GSM8K_TEST_URL}")
    with urllib.request.urlopen(GSM8K_TEST_URL) as response:  # noqa: S310 (trusted URL)
        raw = response.read()

    line_count = raw.count(b"\n")
    print(f"Fetched {len(raw) / 1024:.0f} KiB ({line_count} records). Compressing...")

    # gzip is fine for the write here; the demo reads it back with aiogzip.
    with gzip.open(OUTPUT_PATH, "wb") as f:
        f.write(raw)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
