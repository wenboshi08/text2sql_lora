"""M2 — zero-shot baseline evaluation (run before training; design doc §6/§8 M2).

Scores the un-fine-tuned Qwen2.5-Coder-7B-Instruct on the held-out set
(default data/processed/test.jsonl, a Spider-validation subset), reporting
EM / SQL validity / execution accuracy, with optional --judge for independent
LLM semantic equivalence.

Generation / scoring / aggregation are delegated to src.harness so that the
fine-tuned evaluation (src/eval.py) uses the identical code path.

Usage:
    python -m src.baseline_eval                                     # defaults (4-bit, T4-friendly)
    python -m src.baseline_eval --limit 50                          # smoke test
    python -m src.baseline_eval --db-root data/spider_database      # enable execution eval
    python -m src.baseline_eval --judge --judge-api-key sk-xxx      # add semantic judge (DeepSeek)

Resume: each prediction is written to results/baseline.partial.jsonl immediately;
re-running skips already-completed samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.harness import add_semantic_judge, aggregate, load_records, run_eval
from src.prompt import PROMPT_VERSION

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_base_model(model_name: str, quant: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[baseline] loading tokenizer/model: {model_name} (quant={quant})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16 tensor cores; fp16 is safest
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    model.eval()
    if not torch.cuda.is_available():
        print("[baseline] warning: no GPU detected; CPU inference is very slow (debug with tiny --limit)")
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 zero-shot baseline evaluation")
    ap.add_argument("--test-jsonl", default="data/processed/test.jsonl")
    ap.add_argument("--limit", type=int, default=500, help="number of eval samples (default 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--db-root", default=None, help="Spider database/ dir; enables execution eval")
    ap.add_argument("--out", default="results/baseline.json")
    ap.add_argument("--judge", action="store_true", help="add independent LLM semantic judge")
    ap.add_argument("--judge-api-base", default="https://api.deepseek.com")
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--judge-api-key", default=None, help="or set OPENAI_API_KEY env var")
    ap.add_argument("--judge-sample", type=int, default=200, help="judge sample size (cost control)")
    args = ap.parse_args()

    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        raise SystemExit(f"[baseline] {test_path} not found; run data/prep.py first")
    records = load_records(test_path, args.limit, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.jsonl")
    db_root = Path(args.db_root) if args.db_root else None

    model, tokenizer = load_base_model(args.model, args.quant)
    results = run_eval(model, tokenizer, records, args.max_new_tokens, db_root, partial_path)

    metrics_summary = aggregate(results)
    if args.judge:
        metrics_summary.update(add_semantic_judge(
            records, results,
            api_base=args.judge_api_base, model=args.judge_model,
            api_key=args.judge_api_key, sample=args.judge_sample,
        ))

    report = {
        "model": args.model,
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
    print(f"[baseline] done -> {out_path}")
    print(f"[baseline] metrics: {json.dumps(metrics_summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
