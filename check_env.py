import torch, transformers, datasets, bitsandbytes, peft
print("torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("BF16 supported:", getattr(torch.cuda, "is_bf16_supported", lambda: False)())