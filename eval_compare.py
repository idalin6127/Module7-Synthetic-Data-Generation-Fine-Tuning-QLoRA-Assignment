# eval_compare.py
import os, csv, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

os.makedirs("outputs", exist_ok=True)

BASE = "HuggingFaceH4/zephyr-7b-beta"
FT_DIR = "./models/zephyr-7b-qlora"

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True, trust_remote_code=True)

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", trust_remote_code=True, quantization_config=quant
)
ft_model = PeftModel.from_pretrained(base_model, FT_DIR)

def generate_answer(model, question: str) -> str:
    system = "You are a helpful academic Q&A assistant specialized in scholarly content."
    prompt = f"<|system|>{system}<|user|>{question}<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,                # deterministic for fair comparison
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("<|assistant|>")[-1].strip()

questions = [
    "What was the training objective for this week's dataset?",
    "When generating Q&A from paper abstracts, how should we handle details that do not exist in the paper?",
    "What are the core differences between QLoRA and full fine-tuning?",
    "Why is BF16 recommended on A100 GPUs?",
    "Summarize the key points of the instruction-tuning data format we used.",
    "Give an example of safety/contradiction handling in academic Q&A.",
    "List the main techniques that make Unsloth/QLoRA memory-efficient.",
    "How should we evaluate performance before vs. after fine-tuning?",
    "Why do we enable gradient checkpointing during training?",
    "Why include edge-case questions in the dataset?",
]

rows = []
for q in questions:
    base_ans = generate_answer(base_model, q)
    ft_ans   = generate_answer(ft_model, q)
    rows.append({"question": q, "base_answer": base_ans, "ft_answer": ft_ans})

out_path = "outputs/evaluation.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["question", "base_answer", "ft_answer"])
    w.writeheader(); w.writerows(rows)

print(f"Saved comparison to {out_path}")
