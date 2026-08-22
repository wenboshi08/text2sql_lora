#!/usr/bin/env python3
"""M1 — text-to-SQL data preparation: download / clean / dedupe / split / contamination check.

Corresponds to design doc §3.3 and §8 M1.

Data sources:
  * BIRD  : question/evidence/SQL/db_id from `birdsql/bird23-train-filtered` (official filtered split);
            schema (CREATE TABLE DDL) joined by db_id from `xu3kev/BIRD-SQL-data-train`.
  * Spider: question/SQL/db_id from `xlangai/spider` (train + validation official splits);
            schema from the official Spider `tables.json` (Zenodo; covers all train+dev DBs).

Outputs (--out-dir, default data/processed/):
  * train.jsonl / val.jsonl / test.jsonl — structured records (formatted later via build_messages())
  * meta.json                              — cleaning / split / contamination stats

Record fields:
  {id, db_id, dialect, schema, question, evidence, sql, source}

Cleaning pipeline (in order):
  1. schema normalization (drop SQLite system table sqlite_sequence; Spider compact -> CREATE TABLE DDL)
  2. sqlglot parseability filter (dialect unified to sqlite)
  3. input-length filter (whitespace token count of schema+question+evidence as a token proxy)
  4. global dedup at two levels: (schema, question) and (schema, question, sql)
  5. split + cross-split contamination check (train vs val / train vs test)

Dependencies: pip install datasets sqlglot
Run: python data/prep.py [--limit 200] [--out-dir data/processed]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Config constants
# --------------------------------------------------------------------------- #

BIRD_FILTERED = "birdsql/bird23-train-filtered"          # question/evidence/SQL/db_id
BIRD_SCHEMA   = "xu3kev/BIRD-SQL-data-train"             # embedded schema (CREATE TABLE DDL)
SPIDER        = "xlangai/spider"                         # question/query/db_id (train/validation)
SPIDER_TABLES_URL = "https://zenodo.org/records/5205322/files/tables.json?download=1"
SPIDER_TABLES_CACHE = "data/spider_tables.json"          # local cache of the official tables.json

DIALECT = "sqlite"  # unified dialect: BIRD and Spider are both SQLite


# --------------------------------------------------------------------------- #
# Schema handling
# --------------------------------------------------------------------------- #

_SQLITE_SEQ = re.compile(r"^\s*CREATE\s+TABLE\s+sqlite_sequence\b.*$", re.IGNORECASE | re.MULTILINE)

# Spider type -> SQLite type (SQLite is dynamically typed; this is only for nicer semantics).
_SPIDER_TYPE_MAP = {
    "number": "INTEGER",
    "text": "TEXT",
    "time": "TEXT",
    "boolean": "INTEGER",
    "others": "TEXT",
}


def clean_bird_schema(schema: str) -> str:
    """Drop the SQLite system table mixed into BIRD schemas and collapse blank lines."""
    schema = _SQLITE_SEQ.sub("", schema)
    return re.sub(r"\n{3,}", "\n\n", schema).strip()


def _download_spider_tables(cache_path: str = SPIDER_TABLES_CACHE) -> List[dict]:
    """Download the official Spider tables.json (cached) and return the db list."""
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import urllib.request

    print("[prep] downloading official Spider tables.json ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SPIDER_TABLES_URL, str(path))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[prep] failed to download Spider tables.json: {e}\n"
            f"       download it manually from {SPIDER_TABLES_URL} to {path}"
        ) from e
    return json.loads(path.read_text(encoding="utf-8"))


def _tables_to_ddl_map(tables: List[dict]) -> Dict[str, str]:
    """Convert official Spider tables.json (list of db objects) to {db_id: CREATE TABLE DDL}.

    Uses `*_original` names (the real DB column/table names, which match the gold
    SQL modulo case) and double-quotes every identifier so spaces / reserved words
    / special characters still produce parseable, executable DDL. `table_idx < 0`
    is the "*" pseudo-column and is skipped (0-based table indices).
    """
    schema_map: Dict[str, str] = {}
    for db in tables:
        db_id = db.get("db_id")
        table_names = db.get("table_names_original") or []
        columns = db.get("column_names_original") or []
        types = db.get("column_types") or []

        by_table: Dict[int, List[str]] = {}
        for (tidx, col_name), col_type in zip(columns, types):
            if tidx is None or tidx < 0:
                continue  # skip "*" (table index -1)
            sql_type = _SPIDER_TYPE_MAP.get((col_type or "").lower(), (col_type or "TEXT").upper())
            by_table.setdefault(tidx, []).append(f'"{col_name}" {sql_type}')

        ddl = []
        for tidx, col_defs in sorted(by_table.items()):
            tname = table_names[tidx] if tidx < len(table_names) else f"table_{tidx}"
            ddl.append(f'CREATE TABLE "{tname}" ({", ".join(col_defs)});')
        if ddl:
            schema_map[db_id] = "\n".join(ddl)
    return schema_map


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Dedup normalization: lowercase + whitespace collapse + drop space before punctuation."""
    s = re.sub(r"\s+", " ", (s or "").lower())
    return re.sub(r"\s+([.,;:!?])", r"\1", s).strip()


def is_valid_sql(sql: str, dialect: str = DIALECT) -> bool:
    """Whether the gold SQL parses under sqlglot. Empty/error is invalid."""
    if not sql or not sql.strip():
        return False
    try:
        import sqlglot
        return len(sqlglot.parse(sql, read=dialect)) > 0
    except Exception:  # noqa: BLE001
        return False


def approx_tokens(text: str) -> int:
    """Rough token count via whitespace split (no tokenizer available here).
    The real tokenizer applies the final truncation at training time."""
    return len((text or "").split())


# --------------------------------------------------------------------------- #
# Data loading (returns list[dict])
# --------------------------------------------------------------------------- #

def _load_hf(name: str, split: Optional[str] = None):
    """Lazily load an HF dataset with a clear error message."""
    from datasets import load_dataset
    try:
        if split is None:
            return load_dataset(name)
        return load_dataset(name, split=split)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[prep] failed to load {name} ({split or 'all splits'}): {e}\n"
                         f"       ensure `pip install datasets` and Hugging Face access.") from e


def _build_schema_map(name: str, key_col: str, schema_col: str) -> Dict[str, str]:
    """Build a {db_id: schema} map from a schema-bearing HF mirror (one schema per db).

    Iterates over ALL splits of the mirror because a schema-bearing mirror may
    split schemas across multiple splits.
    """
    ds = _load_hf(name)
    splits = ds.values() if isinstance(ds, dict) else [ds]
    schema_map: Dict[str, str] = {}
    missing = 0
    for split in splits:
        for row in split:
            db_id = row.get(key_col)
            schema = row.get(schema_col) or ""
            if not db_id:
                continue
            if db_id not in schema_map:
                schema = clean_bird_schema(schema)
                if schema:
                    schema_map[db_id] = schema
                else:
                    missing += 1
    if missing:
        print(f"[prep] {name}: {missing} db_ids have no valid schema; skipped")
    return schema_map


def load_bird() -> List[dict]:
    """BIRD: official filtered split (question/evidence/SQL) joined with schema map."""
    print("[prep] loading BIRD ...")
    rows = _load_hf(BIRD_FILTERED, split="train")
    schema_map = _build_schema_map(BIRD_SCHEMA, key_col="db_id", schema_col="schema")

    records: List[dict] = []
    missing_schema = 0
    for i, row in enumerate(rows):
        db_id = row.get("db_id")
        schema = schema_map.get(db_id, "")
        if not schema:
            missing_schema += 1
            continue
        records.append({
            "id": f"bird-{db_id}-{i}",
            "db_id": db_id,
            "dialect": DIALECT,
            "schema": schema,
            "question": (row.get("question") or "").strip(),
            "evidence": (row.get("evidence") or "").strip(),
            "sql": (row.get("SQL") or "").strip(),
            "source": "bird",
        })
    if missing_schema:
        print(f"[prep] BIRD: {missing_schema} records dropped (db_id not matched to a schema)")
    return records


def load_spider() -> List[dict]:
    """Spider: train/validation splits joined with the official tables.json schema map."""
    print("[prep] loading Spider ...")
    ds = _load_hf(SPIDER)  # dict with keys: train, validation
    schema_map = _tables_to_ddl_map(_download_spider_tables())

    records: List[dict] = []
    split_names = [s for s in ("train", "validation") if s in ds]
    if not split_names:
        raise SystemExit("[prep] xlangai/spider has no train/validation split; check the dataset structure")
    for split in split_names:
        missing_schema = 0
        for i, row in enumerate(ds[split]):
            db_id = row.get("db_id")
            schema = schema_map.get(db_id, "")
            if not schema:
                missing_schema += 1
                continue
            records.append({
                "id": f"spider-{split}-{db_id}-{i}",
                "db_id": db_id,
                "dialect": DIALECT,
                "schema": schema,
                "question": (row.get("question") or "").strip(),
                "evidence": "",
                "sql": (row.get("query") or "").strip(),
                "source": f"spider-{split}",
            })
        if missing_schema:
            print(f"[prep] Spider[{split}]: {missing_schema} records dropped (db_id not matched to a schema)")
    return records


# --------------------------------------------------------------------------- #
# Cleaning + dedup
# --------------------------------------------------------------------------- #

def clean(records: List[dict], max_input_tokens: int) -> List[dict]:
    """sqlglot parseability filter + length filter."""
    kept: List[dict] = []
    dropped: Counter = Counter()
    for r in records:
        if not is_valid_sql(r["sql"]):
            dropped["sqlglot_invalid"] += 1
            continue
        n_tok = approx_tokens(r["schema"] + " " + r["question"] + " " + r["evidence"])
        if n_tok > max_input_tokens:
            dropped["too_long"] += 1
            continue
        kept.append(r)
    print(f"[prep] cleaning dropped: {dict(dropped)}")
    return kept


def dedupe(records: List[dict]) -> List[dict]:
    """Two-level dedup: (schema, question) and (schema, question, sql). Keep first occurrence."""
    seen_pq: set = set()
    seen_pqs: set = set()
    kept: List[dict] = []
    dup_pq = dup_pqs = 0
    for r in records:
        pq = (_norm(r["schema"]), _norm(r["question"]))
        pqs = pq + (_norm(r["sql"]),)
        if pq in seen_pq:
            dup_pq += 1
            continue
        if pqs in seen_pqs:
            dup_pqs += 1
            continue
        seen_pq.add(pq)
        seen_pqs.add(pqs)
        kept.append(r)
    print(f"[prep] dedupe: {dup_pq} (schema,question) dups, {dup_pqs} full-triple dups")
    return kept


# --------------------------------------------------------------------------- #
# Split + contamination check + write
# --------------------------------------------------------------------------- #

def _assign_bird_split(records: List[dict], val_ratio: float = 0.10) -> Dict[str, List[dict]]:
    """Deterministic BIRD split by db_id (md5(db_id) into val); stable across runs."""
    train, val = [], []
    for r in records:
        h = int(hashlib.md5(r["db_id"].encode("utf-8")).hexdigest(), 16)
        if h % 1000 < int(val_ratio * 1000):
            val.append(r)
        else:
            train.append(r)
    return {"train": train, "val": val}


def _overlap(a: List[dict], b: List[dict]) -> Dict[str, int]:
    """Cross-split contamination counts (two key levels)."""
    b_pq = {(_norm(x["schema"]), _norm(x["question"])) for x in b}
    b_pqs = {(_norm(x["schema"]), _norm(x["question"]), _norm(x["sql"])) for x in b}
    pq = sum(1 for x in a if (_norm(x["schema"]), _norm(x["question"])) in b_pq)
    pqs = sum(1 for x in a if (_norm(x["schema"]), _norm(x["question"]), _norm(x["sql"])) in b_pqs)
    return {"schema_question_overlap": pq, "full_triple_overlap": pqs}


def split_and_write(records: List[dict], out_dir: Path) -> dict:
    """Assemble train/val/test, write JSONL + meta, run contamination check."""
    out_dir.mkdir(parents=True, exist_ok=True)

    bird = [r for r in records if r["source"] == "bird"]
    spider_train = [r for r in records if r["source"] == "spider-train"]
    spider_val = [r for r in records if r["source"] == "spider-validation"]

    bird_split = _assign_bird_split(bird)

    splits = {
        "train": bird_split["train"] + spider_train,
        "val": bird_split["val"],
        "test": spider_val,  # naturally disjoint from spider_train schemas; used as held-out
    }

    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[prep] {name}.jsonl: {len(rows)} records -> {path}")

    contamination = {
        "train_vs_val": _overlap(splits["train"], splits["val"]),
        "train_vs_test": _overlap(splits["train"], splits["test"]),
    }
    leak = any(v["schema_question_overlap"] > 0 for v in contamination.values())
    if leak:
        print("[prep] WARNING: cross-split (schema, question) leakage detected; check dedup/split logic!")
        print(f"[prep] contamination details: {json.dumps(contamination, indent=2)}")
    else:
        print("[prep] contamination check passed: train has no (schema, question) overlap with val/test")

    meta = {
        "sources": {"bird": BIRD_FILTERED, "spider": SPIDER},
        "dialect": DIALECT,
        "counts": {k: len(v) for k, v in splits.items()},
        "contamination": contamination,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prep] meta.json -> {meta_path}")
    return meta


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="text-to-SQL data preparation (M1)")
    ap.add_argument("--out-dir", default="data/processed", help="output directory")
    ap.add_argument("--max-input-tokens", type=int, default=4000,
                    help="whitespace-token limit for schema+question+evidence (approx; real tokenizer truncates later)")
    ap.add_argument("--skip-bird", action="store_true", help="skip BIRD")
    ap.add_argument("--skip-spider", action="store_true", help="skip Spider")
    ap.add_argument("--limit", type=int, default=None, help="keep only first N records (smoke test)")
    args = ap.parse_args()

    records: List[dict] = []
    if not args.skip_bird:
        records += load_bird()
    if not args.skip_spider:
        records += load_spider()
    if not records:
        raise SystemExit("[prep] no data loaded; check --skip-* flags and network")

    print(f"[prep] raw records (before dedup): {len(records)}")

    records = clean(records, args.max_input_tokens)
    records = dedupe(records)
    if args.limit:
        records = records[: args.limit]
        print(f"[prep] --limit applied; keeping first {args.limit} records")

    meta = split_and_write(records, Path(args.out_dir))
    print(f"[prep] done. split sizes: {meta['counts']}")


if __name__ == "__main__":
    main()
