#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qlora.py — QLoRA fine-tuning for LLaMA 3.1 7B (Unsloth 4-bit, A100-optimized)

Usage:
  python train_qlora.py --jsonl ./data/synthetic_qa.jsonl \
                        --model unsloth/llama-3.1-7b-unsloth-bnb-4bit \
                        --out ./models/llama3-7b-qlora-finetuned \
                        --epochs 2 --batch 8 --grad_accum 2 --lr 2e-4 --bf16 --max_seq_len 4096

Notes:
- Requires: unsloth, transformers, datasets, accelerate, peft, bitsandbytes
- Expects JSONL with a "text" field containing the full chat-formatted sample.
"""

import os
import sys
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, SFTTrainer

# ===== A100/H100: Enable TF32 for higher throughput =====
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# ========================================================


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to synthetic_qa.jsonl")
    ap.add_argument("--model", default="unsloth/llama-3.1-7b-unsloth-bnb-4bit", help="Base model id or path")
    ap.add_argument("--out", required=True, help="Output dir to save LoRA adapters / model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="Per-device train batch size")
    ap.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=str, default="2e-4", help="Learning rate (e.g., 2e-4)")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    ap.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.jsonl):
        print(f"[ERROR] JSONL not found: {args.jsonl}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    print(f"[INFO] Loading dataset from {args.jsonl}")
    dataset = load_dataset("json", data_files=args.jsonl, split="train")

    print(f"[INFO] Loading base model: {args.model}")
    # Load 4-bit model with Unsloth
    model = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,          # let Unsloth/BNB decide
        load_in_4bit=True,
    )

    print("[INFO] Preparing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Prefer FlashAttention2 if available (silently ignore if not present)
    try:
        if getattr(model, "config", None) is not None:
            attn_cfg = getattr(model.config, "attn_config", {}) or {}
            attn_cfg["attn_impl"] = "flash_attention_2"
            model.config.attn_config = attn_cfg
    except Exception:
        pass

    # Save VRAM with gradient checkpointing by disabling past kv cache
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    # Make sure pad_token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Enabling LoRA (QLoRA)")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized ckpt
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("[INFO] Configuring training arguments (A100-optimized)")
    # Auto prefer BF16 on A100 when user didn't force FP16
    auto_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    fp16_flag = bool(args.fp16)
    bf16_flag = bool(args.bf16) or (auto_bf16 and not fp16_flag)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=float(args.lr),

        # Logging & saving
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],

        # === A100 key settings ===
        fp16=fp16_flag,
        bf16=bf16_flag,
        gradient_checkpointing=True,   # save VRAM
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",      # 4-bit friendly optimizer
        max_grad_norm=1.0,
    )

    print("[INFO] Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,           # one sample per sequence; safer for QA formatting
        args=training_args,
    )

    trainer.train()

    print("[INFO] Saving model & tokenizer to:", args.out)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("[OK] Training complete.")


if __name__ == "__main__":
    main()
