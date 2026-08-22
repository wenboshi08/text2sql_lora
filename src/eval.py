"""M4 — fine-tuned model evaluation (design doc §8 M4).

Reuses src.harness (identical generation/scoring/aggregation as baseline) but
loads base + LoRA adapter via PeftModel. Scores the same held-out set, writes
results/finetuned.json, and compares against results/baseline.json to produce
comparison.json.

By default the eval set is aligned to baseline.json's sample ids (apples-to-apples);
without baseline.json it falls back to same-seed/same-limit sampling.

Usage:
    python -m src.eval --adapter outputs/lora
    python -m src.eval --adapter outputs/lora --db-root data/spider_database
    python -m src.eval --adapter outputs/lora --judge --judge-api-key sk-xxx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.harness import add_semantic_judge, aggregate, load_records, run_eval
from src.prompt import PROMPT_VERSION

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_model_with_adapter(base_model: str, adapter_dir: str, quant: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] loading base={base_model} + adapter={adapter_dir} (quant={quant})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def select_matching_baseline(records: List[Dict], baseline_path: Path) -> Optional[List[Dict]]:
    """Align records to baseline.json sample ids (order-preserving). None if unavailable."""
    if not baseline_path.exists():
        return None
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    ids = [s["id"] for s in base.get("samples", [])]
    by_id = {r["id"]: r for r in records}
    matched = [by_id[i] for i in ids if i in by_id]
    if not matched:
        return None
    print(f"[eval] aligned eval set to baseline: {len(matched)}/{len(ids)} samples")
    return matched


def compare_and_write(baseline_path: Path, finetuned: Dict, out_path: Path) -> None:
    """Read baseline metrics and write the delta comparison. Skips if baseline absent."""
    if not baseline_path.exists():
        print(f"[eval] {baseline_path} not found; skipping comparison (writing finetuned only)")
        return
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    bm, fm = base.get("metrics", {}), finetuned.get("metrics", {})
    delta = {}
    for k in ("exact_match", "sql_validity", "execution_accuracy", "semantic_equiv"):
        if k in bm and k in fm and bm[k] is not None and fm[k] is not None:
            delta[k] = round(fm[k] - bm[k], 4)
    comparison = {"baseline": bm, "finetuned": fm, "delta": delta}
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] comparison -> {out_path}")
    print(f"[eval] delta: {json.dumps(delta, ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 fine-tuned model evaluation")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (M3 output)")
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--test-jsonl", default="data/processed/test.jsonl")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--db-root", default=None)
    ap.add_argument("--out", default="results/finetuned.json")
    ap.add_argument("--baseline", default="results/baseline.json")
    ap.add_argument("--comparison", default="results/comparison.json")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-api-base", default="https://api.deepseek.com")
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--judge-sample", type=int, default=200)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"[eval] adapter config not found at {adapter_dir}/adapter_config.json; run M3 first")
    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        raise SystemExit(f"[eval] {test_path} not found; run data/prep.py first")

    records = load_records(test_path, args.limit, args.seed)
    matched = select_matching_baseline(records, Path(args.baseline))
    if matched is not None:
        records = matched

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.jsonl")
    db_root = Path(args.db_root) if args.db_root else None

    model, tokenizer = load_model_with_adapter(args.base_model, str(adapter_dir), args.quant)
    results = run_eval(model, tokenizer, records, args.max_new_tokens, db_root, partial_path)

    metrics_summary = aggregate(results)
    if args.judge:
        metrics_summary.update(add_semantic_judge(
            records, results,
            api_base=args.judge_api_base, model=args.judge_model,
            api_key=args.judge_api_key, sample=args.judge_sample,
        ))

    report = {
        "base_model": args.base_model,
        "adapter": str(adapter_dir),
        "prompt_version": PROMPT_VERSION,
        "dataset": str(test_path),
        "n": len(results),
        "seed": args.seed,
        "quantization": args.quant,
        "metrics": metrics_summary,
        "exec_note": f"db_root={db_root}" if db_root else "no db_root; execution_accuracy not computed",
        "samples": results,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(f"[eval] done -> {out_path}")
    print(f"[eval] metrics: {json.dumps(metrics_summary, ensure_ascii=False)}")

    compare_and_write(Path(args.baseline), report, Path(args.comparison))


if __name__ == "__main__":
    main()
