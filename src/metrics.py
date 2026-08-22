"""Evaluation metrics: exact match / SQL validity / execution accuracy.

Design doc §6:
  * exact match        — lower bound (penalizes different-but-correct SQL)
  * SQL validity       — whether the prediction parses under sqlglot
  * execution accuracy — primary metric; runs against a real SQLite DB
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import List, Optional, Sequence, Tuple

import sqlglot

DIALECT = "sqlite"


# --------------------------------------------------------------------------- #
# Parsing / normalization
# --------------------------------------------------------------------------- #

def parse_one(sql: str, dialect: str = DIALECT) -> Optional[sqlglot.exp.Expression]:
    """Parse a single statement; return None on failure or empty input."""
    if not sql or not sql.strip():
        return None
    try:
        stmts = sqlglot.parse(sql, read=dialect)
        return stmts[0] if stmts else None
    except Exception:  # noqa: BLE001
        return None


def canonical(sql: str, dialect: str = DIALECT) -> Optional[str]:
    """AST canonical form (ignores case/whitespace/quote style). None if unparseable."""
    expr = parse_one(sql, dialect)
    return str(expr) if expr is not None else None


def _norm_str(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower())
    return re.sub(r"\s+([.,;:!?])", r"\1", s).strip()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def sql_validity(pred: str, dialect: str = DIALECT) -> bool:
    return parse_one(pred, dialect) is not None


def exact_match(pred: str, gold: str, dialect: str = DIALECT) -> bool:
    """Normalized exact match: compare AST canonical forms when both parse,
    otherwise fall back to string normalization."""
    cp, cg = canonical(pred, dialect), canonical(gold, dialect)
    if cp is not None and cg is not None:
        return cp == cg
    return _norm_str(pred) == _norm_str(gold)


def _val_key(v) -> str:
    """Normalize a result-set cell: None -> NULL; integral floats 1.0/1 unified;
    everything else lowercased string."""
    if v is None:
        return "NULL"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).lower()


def _rows_key(rows: Sequence[Tuple]) -> List[Tuple[str, ...]]:
    return sorted(tuple(_val_key(v) for v in row) for row in rows)


def _execute(db_path: str, sql: str) -> Optional[List[Tuple]]:
    """Execute a read-only query on a SQLite DB and return rows.

    Returns None for non-query statements, missing DB, or execution errors.
    """
    if not os.path.exists(db_path):
        return None
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return None
    # Only allow read statements (defensive: predictions must not mutate the DB).
    if not re.match(r"^\s*(select|with|pragma|explain)\b", sql, re.IGNORECASE):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            cur = conn.execute(sql)
            if cur.description is None:
                return None
            return cur.fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None


def execution_match(pred: str, gold: str, db_path: Optional[str]) -> Optional[bool]:
    """Execution-level comparison. Returns None (skip sample) when db_path is
    absent or the DB file does not exist."""
    if not db_path or not os.path.exists(db_path):
        return None
    rows_pred = _execute(db_path, pred)
    rows_gold = _execute(db_path, gold)
    if rows_pred is None or rows_gold is None:
        return False
    return _rows_key(rows_pred) == _rows_key(rows_gold)
