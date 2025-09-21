import os, subprocess, sys, json

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return str(e)

print("=== nvidia-smi ===")
print(run("nvidia-smi || true"))

# ---- PyTorch 环境检查 ----
try:
    import torch
    print("\n=== PyTorch / CUDA ===")
    print("torch.version:", torch.__version__)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        dev_name = torch.cuda.get_device_name(0)
        print("device_name:", dev_name)
        print("compute_capability:", torch.cuda.get_device_capability(0))
        print("total_memory(GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
        print("cuda_runtime_version:", torch.version.cuda)

        # BF16 / FP16 / TF32
        print("\n=== Precision support ===")
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        print("bf16_supported:", bf16_ok)
        print("tf32_matmul_enabled_before:", torch.backends.cuda.matmul.allow_tf32)
        # 尝试开启 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("tf32_matmul_enabled_after:", torch.backends.cuda.matmul.allow_tf32)

        # FlashAttention2 可用性（软探测）
        print("\n=== FlashAttention2 ===")
        try:
            import flash_attn_cuda
            print("flash_attn_cuda: AVAILABLE")
        except Exception as e:
            print("flash_attn_cuda: not found (this is OK; Unsloth/HF可自动选择其他实现)")

        # bitsandbytes 检查
        print("\n=== bitsandbytes ===")
        try:
            import bitsandbytes as bnb
            print("bitsandbytes:", bnb.__version__)
            from bitsandbytes.cuda_setup.main import evaluate_cuda_setup
            print("bnb_cuda_setup:", evaluate_cuda_setup())
        except Exception as e:
            print("bitsandbytes import failed:", repr(e))

        # 简单判断 A100 / H100（基于名字）
        dev_lower = dev_name.lower()
        guess = "UNKNOWN"
        if "h100" in dev_lower:
            guess = "H100"
        elif "a100" in dev_lower:
            guess = "A100"
        print("\n=== GPU Guess ===")
        print("By name guess:", guess)

except Exception as e:
    print("\n[ERROR] torch check failed:", repr(e))

    # then run python gpu_check.py
    # if need to install pytorch first, then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # and verify it: python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    # You will see something like: 2.5.1 True... My Gpu plat is2.5.1+cu121 True inference-ai GPU cuda
    # So I need to check more information like Model, computing power (major/minor), video memory
    # run: python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name); print(p.major, p.minor); print('Total Memory (GB):', round(p.total_memory/1024**3,2))"
    # the result is: Device name: inference-ai GPU cuda, Compute capability: 8.6, Total Memory (GB): 47.99. It is A100. (Because if H100, Hopper structure is 9.0)