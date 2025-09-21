#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio demo for Academic Q&A (Base + LoRA)
- Base: HuggingFaceH4/zephyr-7b-beta (4-bit)
- Adapters: ./models/zephyr-7b-qlora
"""
import argparse
import torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
ADAPTER_DIR = "./models/zephyr-7b-qlora"   # change if your path differs
SYSTEM_PROMPT = "You are a helpful academic Q&A assistant specialized in scholarly content."

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,   # A100
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    return tokenizer, model

tokenizer, model = load_model()

def infer(question, max_new_tokens=256, do_sample=False):
    prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>{question}<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split("<|assistant|>")[-1].strip()
    return answer

def build_ui():
    with gr.Blocks(title="Academic Q&A (Fine-tuned via QLoRA)") as demo:
        gr.Markdown("## Academic Q&A (Zephyr-7B + QLoRA)\nAsk any academic-style question. Model runs base-4bit + LoRA adapters.")
        with gr.Row():
            q = gr.Textbox(label="Your Question", placeholder="e.g., Why is BF16 recommended on A100 GPUs?")
        with gr.Row():
            max_tokens = gr.Slider(32, 512, value=256, step=8, label="Max new tokens")
            sampling = gr.Checkbox(value=False, label="Use sampling (turn off for deterministic)")
        btn = gr.Button("Generate")
        a = gr.Textbox(label="Answer")

        btn.click(fn=infer, inputs=[q, max_tokens, sampling], outputs=a)
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (0.0.0.0 to expose)")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(server_name=args.host, server_port=args.port, share=args.share)