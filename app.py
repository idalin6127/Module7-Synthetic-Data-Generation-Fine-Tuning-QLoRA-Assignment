# app.py  —— ultra-low-memory gradio demo
import os, gc, argparse, torch, gradio as gr
from typing import Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEF_BASE = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEF_LORA = os.environ.get("LORA_DIR",  "./models/zephyr-7b-qlora")  # 可为空
DEFAULT_SYSTEM = "You are a helpful academic Q&A assistant specialized in scholarly content."

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEF_BASE, help="HF model id or local path")
    p.add_argument("--lora", default=DEF_LORA, help="Optional LoRA folder")
    p.add_argument("--device", choices=["auto","cpu"], default="auto")
    p.add_argument("--quant", choices=["none","4bit","8bit"], default="4bit",
                   help="GPU: 4bit 最省内存；CPU 请选择 none")
    p.add_argument("--max-gpu-mem", default=None, help='限制 GPU 显存 e.g. "6GiB"')
    p.add_argument("--offload", action="store_true", help="必要时下放到 CPU/磁盘以避免 OOM")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()

def build_quant(quant: str) -> Optional[BitsAndBytesConfig]:
    if not torch.cuda.is_available():
        return None
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # A100/H100 友好
        )
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

def build_max_memory(limit: Optional[str]) -> Optional[Dict]:
    if not (torch.cuda.is_available() and limit):
        return None
    # device 0：限制可用显存，剩余自动 offload 到 CPU（非常省内存，但速度会慢）
    return {0: str(limit), "cpu": "48GiB"}

def load_pipeline(base, lora, device, quant, max_gpu_mem, offload):
    use_cuda = torch.cuda.is_available() and device == "auto"
    quant_cfg = build_quant(quant if use_cuda else "none")
    max_mem   = build_max_memory(max_gpu_mem)

    kwargs = dict(
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if use_cuda else None),
        device_map=("auto" if use_cuda else "cpu"),
    )
    if quant_cfg: kwargs["quantization_config"] = quant_cfg
    if max_mem:   kwargs["max_memory"] = max_mem
    if offload:
        os.makedirs("./offload", exist_ok=True)
        kwargs["offload_folder"] = "./offload"

    print(f"[INFO] loading base={base}, device={kwargs['device_map']}, quant={quant}, max_mem={max_gpu_mem}, offload={offload}")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

    # 尝试挂 LoRA（不存在就跳过）
    if lora and os.path.isdir(lora) and any(os.scandir(lora)):
        try:
            from peft import PeftModel
            print(f"[INFO] attaching LoRA from {lora}")
            model = PeftModel.from_pretrained(model, lora)
        except Exception as e:
            print(f"[WARN] LoRA not attached: {e}")

    # 进一步省显存（禁用缓存 + 启用推理模式）
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True  # 推理中开启 cache 能省算力（显存略增，通常可接受）
    return tok, model

@torch.inference_mode()
def generate_answer(system, user, temperature, top_p, max_new_tokens):
    prompt = f"<|system|>{system.strip()}<|user|>{user.strip()}<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=(float(temperature) > 0),
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # 释放中间张量
    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return text.split("<|assistant|>")[-1].strip()

def build_ui(port, share):
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("## Ultra-Light Academic Q&A Demo (Gradio) — memory-friendly")
        sys_box = gr.Textbox(value=DEFAULT_SYSTEM, label="System", lines=2)
        user_box = gr.Textbox(lines=8, label="Your question")
        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")
            top_p       = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
            max_new     = gr.Slider(16, 512, value=128, step=16, label="Max new tokens")  # 默认 128 更省显存
        out_box = gr.Textbox(lines=12, label="Answer")
        run_btn = gr.Button("Generate", variant="primary")
        run_btn.click(generate_answer, [sys_box, user_box, temperature, top_p, max_new], out_box)
    demo.launch(server_port=port, share=share)

if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = load_pipeline(
        base=args.base, lora=args.lora, device=args.device,
        quant=args.quant, max_gpu_mem=args.max_gpu_mem, offload=args.offload,
    )
    build_ui(args.port, args.share)
