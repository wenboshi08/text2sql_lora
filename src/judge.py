"""LLM-based semantic-equivalence judge (lightweight port of junmingg's judge.py).

Uses an LLM independent from the model under evaluation (OpenAI-compatible
endpoint; DeepSeek by default, Qwen optional) to decide whether a candidate SQL
is semantically equivalent to the gold SQL, avoiding self-preference bias
(design doc §6).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

JUDGE_PROMPT = """You are a SQL equivalence judge. Given a database schema, a question, a gold SQL and a candidate SQL, decide whether the candidate is semantically equivalent to the gold (both produce the same answer for the question).

### Database schema:
{schema}

### Question:
{question}

### Gold SQL:
{gold}

### Candidate SQL:
{candidate}

Is the candidate semantically equivalent to the gold? Reply with exactly one token: YES or NO."""


def _make_client(api_base: str, api_key: str):
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key or key == "EMPTY":
        raise ValueError(
            "No API key provided. Set the --judge-api-key arg or OPENAI_API_KEY env var."
        )
    return OpenAI(api_key=key, base_url=api_base)


def judge_one(
    record: Dict,
    pred: str,
    *,
    api_base: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Optional[bool]:
    """Judge a single sample. Returns None on failure (not counted in metrics)."""
    client = _make_client(api_base, api_key or "")
    content = JUDGE_PROMPT.format(
        schema=record.get("schema", ""),
        question=record.get("question", ""),
        gold=record.get("sql", ""),
        candidate=pred,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=8,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[judge] call failed (sample skipped): {e}")
        return None


def judge_batch(
    records: List[Dict],
    preds: List[str],
    *,
    api_base: str,
    model: str,
    api_key: Optional[str],
    sample: Optional[int] = None,
    seed: int = 42,
) -> List[Optional[bool]]:
    """Judge in batch. `sample` limits the number of API calls (None = all)."""
    idxs = list(range(len(records)))
    if sample is not None and sample < len(idxs):
        import random

        rng = random.Random(seed)
        idxs = rng.sample(idxs, sample)
    results: List[Optional[bool]] = [None] * len(records)
    for i, j in enumerate(idxs):
        results[j] = judge_one(records[j], preds[j], api_base=api_base, model=model, api_key=api_key)
        if (i + 1) % 25 == 0:
            done = [r for r in results if r is not None]
            print(f"[judge] {i + 1}/{len(idxs)} judged, current equiv rate {sum(done) / len(done):.3f}")
    return results
