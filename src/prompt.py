"""Prompt construction — the single source of truth for chat templates.

baseline_eval / train / eval must all go through here so that the base model and
the fine-tuned model use the exact same template and decoding parameters
(junmingg's build_messages() pattern; see design doc §4).
"""

from __future__ import annotations

import re
from typing import Dict, List

# Matches the template in design doc §4.
SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "output a single valid SQLite query and nothing else."
)

PROMPT_VERSION = "1"


def build_messages(record: Dict, include_sql: bool = True) -> List[Dict[str, str]]:
    """Build ChatML messages from a record.

    Expected fields: schema / question / evidence (optional) / sql (optional).
    """
    evidence = (record.get("evidence") or "").strip() or "None"
    user = (
        "### Database schema:\n"
        f"{record['schema']}\n\n"
        "### External knowledge (evidence):\n"
        f"{evidence}\n\n"
        "### Question:\n"
        f"{record['question']}\n\n"
        "### SQL:\n"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if include_sql and record.get("sql"):
        messages.append({"role": "assistant", "content": record["sql"]})
    return messages


_FENCE_LANGS = {"sql", "sqlite", "mysql", "postgres", "postgresql", "psql",
                "text", "plaintext", "code", "python", "bash", "sh", ""}


def clean_sql(text: str) -> str:
    """Strip code fences and "SQL:" labels, returning the cleanest SQL text."""
    if not text:
        return ""
    t = text.strip()
    # Remove ```sql ... ``` fences (take the body before the last fence pair).
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[-2] if len(parts) >= 3 else parts[-1]
        # Drop the first line if it is a language tag (sql / sqlite / ...).
        lines = t.splitlines()
        if len(lines) > 1 and lines[0].strip().lower() in _FENCE_LANGS:
            lines = lines[1:]
        t = "\n".join(lines).strip()
    # Drop standalone "SQL:" label lines.
    lines = [ln for ln in t.splitlines() if not re.match(r"^\s*sql\s*:\s*$", ln, re.IGNORECASE)]
    return "\n".join(lines).strip()


def first_parseable(text: str, dialect: str = "sqlite") -> str:
    """Return the first sqlglot-parseable statement in `text`.

    Tolerates a model that appends an explanation after the SQL. Falls back to
    the first `;`-terminated statement, then to the original text (scored as
    invalid by the metrics layer).
    """
    from src.metrics import parse_one  # Deferred import to avoid a cycle.

    if parse_one(text, dialect):
        return text
    if ";" in text:
        head = text.split(";")[0] + ";"
        if parse_one(head, dialect):
            return head
    return text
