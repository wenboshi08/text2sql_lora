#!/usr/bin/env python3
"""Download the official Spider `database/` directory (needed for execution eval).

The official Spider data package is distributed via a Google Drive direct link
documented in the magesql README (gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J ->
spider_data.zip), which contains database/ (train+dev SQLite files) and
test_database/. This script extracts database/ into --out-dir.

Usage:
    python data/download_spider_dbs.py
    python data/download_spider_dbs.py --out-dir data/spider_database

Fallback: if gdown/network is unavailable, download the Spider package manually
from https://yale-lily.github.io/spider and place database/ at --out-dir.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def _find_database_dir(root: Path) -> Path:
    """Locate the `database` directory inside the extracted tree (contains many subdirs)."""
    candidates = [p for p in root.rglob("database") if p.is_dir()]
    for cand in candidates:
        subdirs = [c for c in cand.iterdir() if c.is_dir()]
        if len(subdirs) >= 50:  # Spider has ~200 databases
            return cand
    if candidates:
        return candidates[0]
    raise SystemExit(f"[spider-dbs] no database/ directory found under {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description="download official Spider databases")
    ap.add_argument("--out-dir", default="data/spider_database")
    ap.add_argument("--work-dir", default="data/_spider_download", help="temp extraction dir")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    zip_path = work_dir / "spider_data.zip"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise SystemExit("[spider-dbs] gdown required: pip install gdown")

    print(f"[spider-dbs] downloading Spider package (Google Drive id={SPIDER_DRIVE_ID}) ...")
    gdown.download(id=SPIDER_DRIVE_ID, output=str(zip_path), quiet=False)

    print(f"[spider-dbs] extracting -> {work_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    db_dir = _find_database_dir(work_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(db_dir, out_dir)
    print(f"[spider-dbs] database/ copied to {out_dir} ({len(list(out_dir.iterdir()))} databases)")

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print("[spider-dbs] done")


if __name__ == "__main__":
    main()
