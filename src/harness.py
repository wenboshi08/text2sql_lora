"""Shared evaluation harness — baseline and fine-tuned models run the exact same
generation / scoring / aggregation code path.

Design doc §6: fair comparison (baseline-first, same template, same decoding
parameters) is guaranteed by this module.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src import metrics
from src.prompt import build_messages, clean_sql, first_parseable


def load_records(path: Path, limit: Optional[int], seed: int) -> List[Dict]:
    """Load a JSONL file and (optionally) sample deterministically."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if limit is not None and limit < len(rows):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(rows)), limit))
        rows = [rows[i] for i in idxs]
        print(f"[harness] deterministically sampled {limit} records (seed={seed})")
    return rows


def db_path_for(record: Dict, db_root: Optional[Path]) -> Optional[str]:
    """Locate the SQLite file for a record's db_id."""
    if not db_root:
        return None
    db_id = record.get("db_id", "")
    for cand in (db_root / db_id / f"{db_id}.sqlite", db_root / f"{db_id}.sqlite"):
        if cand.exists():
            return str(cand)
    return None


def generate(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    """Greedy decoding (reproducible), decoupled from the prompt template."""
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    out = gen[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(out, skip_special_tokens=True)


def run_eval(
    model,
    tokenizer,
    records: List[Dict],
    max_new_tokens: int,
    db_root: Optional[Path],
    partial_path: Path,
) -> List[Dict]:
    """Generate and score sample-by-sample with resume support (partial JSONL)."""
    done: Dict[str, str] = {}
    if partial_path.exists():
        for line in partial_path.open(encoding="utf-8"):
            done.update(json.loads(line))
        print(f"[harness] resuming: {len(done)} records already done")

    results: List[Dict] = []
    t0 = time.time()
    with partial_path.open("a", encoding="utf-8") as pf:
        for i, rec in enumerate(records):
            rec_id = rec["id"]
            if rec_id in done:
                pred = done[rec_id]
            else:
                pred = clean_sql(generate(model, tokenizer, build_messages(rec, include_sql=False),
                                          max_new_tokens))
                pf.write(json.dumps({rec_id: pred}, ensure_ascii=False) + "\n")
                pf.flush()

            sql = first_parseable(pred)
            gold = rec.get("sql", "")
            db_path = db_path_for(rec, db_root)
            results.append({
                "id": rec_id,
                "db_id": rec.get("db_id", ""),
                "question": rec.get("question", ""),
                "gold": gold,
                "pred": pred,
                "em": metrics.exact_match(sql, gold),
                "valid": metrics.sql_validity(sql),
                "exec": metrics.execution_match(sql, gold, db_path),
            })
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"[harness] {i + 1}/{len(records)} ({el / (i + 1):.1f}s/record)")
    return results


def aggregate(results: List[Dict]) -> Dict:
    """Aggregate metrics. execution_accuracy is None when no DB was provided."""
    if not results:
        return {"exact_match": 0.0, "sql_validity": 0.0, "execution_accuracy": None}
    em = sum(1 for r in results if r["em"]) / len(results)
    valid = sum(1 for r in results if r["valid"]) / len(results)
    exec_res = [r["exec"] for r in results if r["exec"] is not None]
    out = {"exact_match": round(em, 4), "sql_validity": round(valid, 4)}
    out["execution_accuracy"] = round(sum(exec_res) / len(exec_res), 4) if exec_res else None
    return out


def add_semantic_judge(records: List[Dict], results: List[Dict], **judge_kwargs) -> Dict:
    """Run the independent LLM semantic judge and append to metrics + samples."""
    from src.judge import judge_batch

    print("[harness] running independent LLM semantic judge ...")
    judg = judge_batch(records, [r["pred"] for r in results], **judge_kwargs)
    for r, j in zip(results, judg):
        r["semantic_equiv"] = j
    judged = [j for j in judg if j is not None]
    if judged:
        return {"semantic_equiv": round(sum(judged) / len(judged), 4)}
    return {}
