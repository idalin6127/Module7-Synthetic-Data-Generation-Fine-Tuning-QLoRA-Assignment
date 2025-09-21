#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qlora_hf.py — QLoRA fine-tuning without Unsloth (Transformers + PEFT + BitsAndBytes)

Usage:
  python train_qlora_hf.py --jsonl ./data/synthetic_qa.jsonl \
    --model unsloth/llama-3.1-7b-unsloth-bnb-4bit \
    --out ./llama3-7b-qlora-finetuned \
    --epochs 2 --batch 8 --grad_accum 2 --lr 2e-4 --bf16 --max_seq_len 4096
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"   # 保险起见，也关掉 JAX/Flax

import os, sys, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === A100: Enable TF32 for higher throughput ===
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# ===============================================


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="unsloth/llama-3.1-7b-unsloth-bnb-4bit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is required for practical QLoRA fine-tuning.")
        sys.exit(3)
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    # 1) Load dataset (expects a "text" field)
    print(f"[INFO] Loading dataset from {args.jsonl}")
    raw_ds = load_dataset("json", data_files=args.jsonl, split="train")

    # 2) Tokenizer
    print("[INFO] Preparing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token  # for causal LM
    pad_id = tokenizer.pad_token_id

    # 3) Tokenize dataset (HF 路线需要这一步)
    def _tok_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=False,
        )
        return out

    print("[INFO] Tokenizing dataset ...")
    ds = raw_ds.map(_tok_fn, batched=True, remove_columns=raw_ds.column_names)

    # 4) 4-bit quantized base model
    print(f"[INFO] Loading base model: {args.model}")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100: prefer BF16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prefer FlashAttention2 if available
    try:
        attn_cfg = getattr(model.config, "attn_config", {}) or {}
        attn_cfg["attn_impl"] = "flash_attention_2"
        model.config.attn_config = attn_cfg
    except Exception:
        pass

    # Pin pad token on model config
    if getattr(model.config, "pad_token_id", None) is None and pad_id is not None:
        model.config.pad_token_id = pad_id

    # 5) Prepare QLoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 配合梯度检查点，关闭 past cache（省显存）
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # 6) Collator（动态 padding，labels = input_ids）
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7) Training args（A100 友好）
    #   - 明确用 BF16（或命令行传 --fp16）
    auto_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    fp16_flag = bool(args.fp16)
    bf16_flag = bool(args.bf16) or (auto_bf16 and not fp16_flag)

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,

        # logging / saving
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],

        # A100 key settings
        fp16=fp16_flag,
        bf16=bf16_flag,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,

        # HF route 要求我们自己做过 tokenization
        remove_unused_columns=False,
    )

    # 8) Train
    print("[INFO] Starting training ...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 9) Save
    print("[INFO] Saving model & tokenizer to:", args.out)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("[OK] Done.")


if __name__ == "__main__":
    main()
