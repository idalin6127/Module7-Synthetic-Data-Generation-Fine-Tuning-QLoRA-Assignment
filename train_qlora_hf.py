#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qlora_hf.py — QLoRA fine-tuning without Unsloth (Transformers + PEFT + BitsAndBytes)

Usage:
  python train_qlora_hf.py --jsonl ./data/synthetic_qa.jsonl \
    --model unsloth/llama-3.1-7b-unsloth-bnb-4bit \
    --out ./llama3-7b-qlora-finetuned \
    --epochs 2 --batch 4 --grad_accum 4 --lr 2e-4 --fp16
"""
import os, sys, argparse, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="unsloth/llama-3.1-7b-unsloth-bnb-4bit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    return ap.parse_args()

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is required for practical QLoRA fine-tuning."); sys.exit(3)
    os.makedirs(args.out, exist_ok=True)

    ds = load_dataset("json", data_files=args.jsonl, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "right"; tokenizer.truncation_side = "right"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if (args.fp16 or not args.bf16) else torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="epoch",
        fp16=bool(args.fp16),
        bf16=bool(args.bf16) and not bool(args.fp16),
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",
        report_to=[],
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=ds,
                      data_collator=collator, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(args.out); tokenizer.save_pretrained(args.out)
    print("[OK] Done. Saved to:", args.out)

if __name__ == "__main__":
    main()
