"""M3 — QLoRA SFT training (design doc §5 / §8 M3).

Method: Unsloth 4-bit + LoRA r=16 (all linear layers) + transformers.Trainer,
with completion-only loss (custom collator: only the assistant SQL part counts
toward the loss). Falls back to plain PEFT when unsloth is unavailable.

No TRL dependency: `DataCollatorForCompletionOnlyLM` was removed from recent
transformers and TRL's SFTTrainer API drifts across versions, so the mask is
computed manually (find the ChatML `<|im_start|>assistant` anchor and set
everything before it to -100), which is stable across transformers versions.

Usage (Colab; run data/prep.py first):
    pip install unsloth peft
    python -m src.train                                          # defaults
    python -m src.train --limit 100 --max-steps 5               # smoke test
    python -m src.train --batch-size 1 --max-seq-length 1536    # low-VRAM (T4)

Outputs (--out-dir, default outputs/lora):
    adapter_model.safetensors + adapter_config.json + tokenizer files
    train_config.json   — training config (reproducibility)
    history.json        — train/val loss curve
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.prompt import PROMPT_VERSION, build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"  # ChatML assistant header (completion-only anchor)

# Qwen linear modules (LoRA targets).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# Model loading (Unsloth preferred, PEFT fallback)
# --------------------------------------------------------------------------- #

def load_model_unsloth(model_name: str, max_seq_length: int):
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=None,                 # auto (A100->bf16, T4->fp16)
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer, is_bfloat16_supported()


def load_model_peft(model_name: str, max_seq_length: int):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    bf16 = torch.cuda.get_device_capability()[0] >= 8 if torch.cuda.is_available() else False
    return model, tokenizer, bf16


# --------------------------------------------------------------------------- #
# Dataset (ChatML text -> tokenized Dataset with labels)
# --------------------------------------------------------------------------- #

def format_texts(tokenizer, path: Path, limit: Optional[int]) -> List[str]:
    """Format records into full ChatML text (with assistant SQL), return text list."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]
    texts = []
    for r in rows:
        msgs = build_messages(r, include_sql=True)
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    print(f"[train] dataset size: {len(texts)} records (from {path})")
    return texts


def _find_suffix_end(ids: List[int], sub: List[int]) -> Optional[int]:
    """Return the end index of the last occurrence of `sub` in `ids` (mask start).
    Returns None if not found."""
    n, m = len(ids), len(sub)
    for i in range(n - m, -1, -1):
        if ids[i:i + m] == sub:
            return i + m
    return None


def tokenize_dataset(tokenizer, texts: List[str], max_seq_length: int):
    """Tokenize + length-filter + completion-only labels. Returns (Dataset, dropped)."""
    from datasets import Dataset

    resp_ids = tokenizer(RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]
    encs = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    rows: List[Dict] = []
    dropped = 0
    for ids, attn in zip(encs["input_ids"], encs["attention_mask"]):
        if len(ids) > max_seq_length:
            dropped += 1
            continue
        start = _find_suffix_end(ids, resp_ids)
        if start is None:
            # Should not happen (format always includes the assistant header); safe fallback.
            labels = [-100] * len(ids)
        else:
            labels = [-100] * start + ids[start:]
        rows.append({"input_ids": ids, "attention_mask": attn, "labels": labels})
    if dropped:
        print(f"[train] dropped {dropped} over-length samples (> {max_seq_length} tokens)")
    return Dataset.from_list(rows), dropped


class CompletionOnlyCollator:
    """Right-pads a batch to equal length; -100 label positions are ignored in loss."""

    def __init__(self, tokenizer):
        # pad_token_id may be 0 (falsy); use an is-not-None check.
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="M3 QLoRA SFT training")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train-jsonl", default="data/processed/train.jsonl")
    ap.add_argument("--val-jsonl", default="data/processed/val.jsonl")
    ap.add_argument("--out-dir", default="outputs/lora")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch_size * grad_accum (default 16)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="use only first N records (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test)")
    ap.add_argument("--no-unsloth", action="store_true", help="force plain PEFT path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path, val_path = Path(args.train_jsonl), Path(args.val_jsonl)
    if not train_path.exists():
        raise SystemExit(f"[train] {train_path} not found; run data/prep.py first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model + tokenizer ----
    use_unsloth = False
    if not args.no_unsloth:
        try:
            model, tokenizer, bf16 = load_model_unsloth(args.model, args.max_seq_length)
            use_unsloth = True
            print("[train] using Unsloth accelerated path")
        except Exception as e:  # noqa: BLE001
            print(f"[train] unsloth unavailable ({e}); falling back to plain PEFT")
    if not use_unsloth:
        model, tokenizer, bf16 = load_model_peft(args.model, args.max_seq_length)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[train] trainable params: {trainable:,} / {total:,} = {trainable / total:.2%}")

    # ---- data ----
    train_texts = format_texts(tokenizer, train_path, args.limit)
    if not train_texts:
        raise SystemExit("[train] training set is empty; check the data file")
    train_ds, _ = tokenize_dataset(tokenizer, train_texts, args.max_seq_length)
    if len(train_ds) == 0:
        raise SystemExit("[train] training set empty after length filter; lower --max-seq-length or check data")
    val_texts = format_texts(tokenizer, val_path, None) if val_path.exists() else []
    val_ds, _ = tokenize_dataset(tokenizer, val_texts, args.max_seq_length) if val_texts else (None, 0)

    # ---- training args ----
    from transformers import Trainer, TrainingArguments

    training_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_8bit",
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    if args.max_steps is not None:
        # Only pass max_steps when explicitly set: transformers 4.5x Trainer.__init__
        # still has an `args.max_steps > 0` check that TypeErrors on None.
        training_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CompletionOnlyCollator(tokenizer),
    )

    print(f"[train] starting: {len(train_ds)} records, effective batch={args.batch_size * args.grad_accum}, "
          f"steps/epoch~={len(train_ds) // (args.batch_size * args.grad_accum)}")
    t0 = time.time()
    trainer.train()
    print(f"[train] training took {(time.time() - t0) / 60:.1f} min")

    # ---- save ----
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    config = {
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "unsloth": use_unsloth,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": TARGET_MODULES,
        "bf16": bf16,
        "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds else 0,
        "seed": args.seed,
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[train] adapter saved -> {out_dir}")
    print(f"[train] config -> {out_dir / 'train_config.json'}, history -> {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
