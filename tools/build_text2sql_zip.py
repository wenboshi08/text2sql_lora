#!/usr/bin/env python3
"""Package the text2sql runtime source into text2sql.zip (for Colab upload).

Contents (relative to repo root; the zip root is the project root, so after
extraction /content/text2sql/ is the project directory):
    data/prep.py
    data/download_spider_dbs.py
    src/__init__.py + src/{prompt,metrics,judge,harness,baseline_eval,train,eval}.py
    requirements.txt
    docs/TEXT2SQL_FINETUNE_DESIGN.md (reference)

Usage: python tools/build_text2sql_zip.py   # produces ./text2sql.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "text2sql.zip"

RUNTIME_FILES = [
    "data/prep.py",
    "data/download_spider_dbs.py",
    "data/download_bird_dev_dbs.py",
    "src/__init__.py",
    "src/prompt.py",
    "src/metrics.py",
    "src/judge.py",
    "src/harness.py",
    "src/baseline_eval.py",
    "src/train.py",
    "src/eval.py",
    "requirements.txt",
    "docs/TEXT2SQL_FINETUNE_DESIGN.md",
]


def main() -> None:
    missing = [f for f in RUNTIME_FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in RUNTIME_FILES:
            zf.write(ROOT / rel, arcname=rel)
    size_kb = OUT.stat().st_size / 1024
    print(f"generated {OUT} ({size_kb:.1f} KB, {len(RUNTIME_FILES)} files)")
    with zipfile.ZipFile(OUT) as zf:
        for name in zf.namelist():
            print("  ", name)


if __name__ == "__main__":
    main()
