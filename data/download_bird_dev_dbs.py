#!/usr/bin/env python3
"""Download the BIRD dev databases (SQLite) for execution-level evaluation.

The official Mini-Dev package (Google Drive, id 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG)
contains the SQLite databases of the 11 BIRD dev databases. This script downloads
it via gdown, extracts the sqlite files into a `{db_id}/{db_id}.sqlite` layout
under --out-dir, ready for `src.eval --db-root`.

Usage:
    python data/download_bird_dev_dbs.py
    python data/download_bird_dev_dbs.py --out-dir data/bird_dev_database

Fallback: if gdown/network is unavailable, download the Mini-Dev package manually
from the badge link in https://github.com/bird-bench/mini_dev and place the
`{db_id}/{db_id}.sqlite` files at --out-dir.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

BIRD_DEV_DRIVE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"


def _collect_sqlite(root: Path):
    """Yield the real `{db_id}/{db_id}.sqlite` files, skipping __MACOSX junk."""
    found = []
    for p in root.rglob("*.sqlite"):
        parts = p.parts
        if "__MACOSX" in parts:
            continue
        if p.name.startswith("._"):
            continue
        # prefer the {db_id}/{db_id}.sqlite layout
        found.append(p)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="download BIRD dev SQLite databases")
    ap.add_argument("--out-dir", default="data/bird_dev_database")
    ap.add_argument("--work-dir", default="data/_bird_dev_download", help="temp extraction dir")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    zip_path = work_dir / "bird_dev_dbs.zip"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        raise SystemExit("[bird-dbs] gdown required: pip install gdown")

    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"[bird-dbs] reusing cached {zip_path}")
    else:
        print(f"[bird-dbs] downloading Mini-Dev package (Google Drive id={BIRD_DEV_DRIVE_ID}, ~800MB) ...")
        gdown.download(id=BIRD_DEV_DRIVE_ID, output=str(zip_path), quiet=False)

    print(f"[bird-dbs] extracting -> {work_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    sqlite_files = _collect_sqlite(work_dir)
    if not sqlite_files:
        raise SystemExit(f"[bird-dbs] no *.sqlite files found under {work_dir}")
    print(f"[bird-dbs] found {len(sqlite_files)} sqlite databases")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for p in sqlite_files:
        db_id = p.stem
        dest = out_dir / db_id
        dest.mkdir(exist_ok=True)
        shutil.copy2(p, dest / f"{db_id}.sqlite")
    print(f"[bird-dbs] databases copied to {out_dir} ({len(list(out_dir.iterdir()))} db_ids)")

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print("[bird-dbs] done")


if __name__ == "__main__":
    main()
