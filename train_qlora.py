
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qlora.py — QLoRA fine-tuning for LLaMA 3.1 7B (Unsloth 4-bit)

Usage:
  python train_qlora.py --jsonl ./data/synthetic_qa.jsonl \
                        --model unsloth/llama-3.1-7b-unsloth-bnb-4bit \
                        --out ./llama3-7b-qlora-finetuned \
                        --epochs 2 --batch 4 --grad_accum 4 --lr 2e-4 --fp16

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to synthetic_qa.jsonl")
    ap.add_argument("--model", default="unsloth/llama-3.1-7b-unsloth-bnb-4bit", help="Base model id or path")
    ap.add_argument("--out", required=True, help="Output dir to save LoRA adapters / model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4, help="Per-device train batch size")
    ap.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=str, default="2e-4", help="Learning rate (e.g., 2e-4)")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    ap.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    ap.add_argument("--max_seq_len", type=int, default=2048)
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
    # Load 4-bit model
    model = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,             # auto
        load_in_4bit=True,
    )

    print("[INFO] Preparing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    print("[INFO] Enabling LoRA (QLoRA)")
    # Typical LLaMA target modules
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("[INFO] Configuring trainer")
    fp16 = bool(args.fp16)
    bf16 = bool(args.bf16) and not fp16  # prefer fp16 if both are given

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=float(args.lr),
        logging_steps=50,
        save_strategy="epoch",
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
        report_to=[],
    )

    print("[INFO] Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,                # keep one sample per sequence
        args=training_args,
    )

    trainer.train()
    print("[INFO] Saving model & tokenizer to:", args.out)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("[OK] Training complete.")

if __name__ == "__main__":
    main()
