#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — One-command runner for the Week7 pipeline.

Subcommands:
  enrich   -> use Week4 meta.jsonl to fetch abstracts (abstracts.json)
  topup    -> pull extra abstracts & merge to 100 (abstracts_100.json)
  qas      -> generate Q&A via OpenAI (papers_qas.json)
  jsonl    -> convert Q&A to training JSONL (synthetic_qa.jsonl)
  train    -> QLoRA fine-tuning
  eval     -> compare base vs fine-tuned and export CSV
  all      -> run everything in sequence

Reads defaults from configs/week7.yaml if present. CLI flags override config.
"""
import os, sys, subprocess, argparse

def load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "week7.yaml")
    data = {}
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print("[warn] Failed to parse configs/week7.yaml:", e)
    return data

def sh(cmd):
    print("+", " ".join(cmd))
    cp = subprocess.run(cmd, check=False)
    if cp.returncode != 0:
        sys.exit(cp.returncode)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    cfg = load_cfg()
    DATA_ROOT   = cfg.get("DATA_ROOT", "../data")
    WEEK4_META  = cfg.get("WEEK4_META", f"{DATA_ROOT}/interim/week4/meta.jsonl")
    OUT_DIR     = cfg.get("OUT_DIR", "./data")
    ABSTRACTS_JSON      = cfg.get("ABSTRACTS_JSON", f"{OUT_DIR}/abstracts.json")
    TOPUP_JSON          = cfg.get("TOPUP_JSON", f"{OUT_DIR}/topup_abstracts.json")
    ABSTRACTS_100_JSON  = cfg.get("ABSTRACTS_100_JSON", f"{OUT_DIR}/abstracts_100.json")
    PAPERS_QAS_JSON     = cfg.get("PAPERS_QAS_JSON", f"{OUT_DIR}/papers_qas.json")
    SYNTH_QA_JSONL      = cfg.get("SYNTH_QA_JSONL", f"{OUT_DIR}/synthetic_qa.jsonl")
    PER_PAPER_QA        = int(cfg.get("PER_PAPER_QA", 5))
    OPENAI_MODEL        = cfg.get("OPENAI_MODEL", "gpt-4o")
    MODEL_NAME          = cfg.get("MODEL_NAME", "unsloth/llama-3.1-7b-unsloth-bnb-4bit")
    OUTPUT_DIR          = cfg.get("OUTPUT_DIR", "./llama3-7b-qlora-finetuned")

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enrich","topup","qas","jsonl","train","eval","all"], help="Pipeline step")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--week4-meta", default=WEEK4_META)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--abstracts", default=ABSTRACTS_JSON)
    ap.add_argument("--topup", default=TOPUP_JSON)
    ap.add_argument("--abstracts100", default=ABSTRACTS_100_JSON)
    ap.add_argument("--papers-qas", default=PAPERS_QAS_JSON)
    ap.add_argument("--jsonl", default=SYNTH_QA_JSONL)
    ap.add_argument("--per-paper", type=int, default=PER_PAPER_QA)
    ap.add_argument("--openai-model", default=OPENAI_MODEL)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out-model", default=OUTPUT_DIR)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    here = os.path.dirname(__file__)
    P = lambda name: os.path.join(here, name)

    if args.cmd in ("enrich","all"):
        sh([sys.executable, P("enrich_with_abstracts.py"),
            "--meta", args.week4_meta, "--out", args.abstracts])

    if args.cmd in ("topup","all"):
        sh([sys.executable, P("top_up_abstracts.py"),
            "--existing", args.abstracts, "--cats", "cs.CL", "cs.AI", "cs.LG",
            "--need", "50", "--out", args.topup])
        sh([sys.executable, P("merge_abstracts.py"),
            "--a", args.abstracts, "--b", args.topup,
            "--out", args.abstracts100, "--limit", "100"])

    if args.cmd in ("qas","all"):
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: please set OPENAI_API_KEY before running 'qas' or 'all'")
            sys.exit(2)
        sh([sys.executable, P("generate_qas_from_abstracts.py"),
            "--input", args.abstracts100, "--out", args.papers_qas,
            "--model", args.openai_model, "--per_paper", str(args.per_paper)])

    if args.cmd in ("jsonl","all"):
        sh([sys.executable, P("generate_jsonl_from_qas.py"),
            "--input", args.papers_qas, "--out", args.jsonl])

    if args.cmd in ("train","all"):
        sh([sys.executable, P("train_qlora.py"),
            "--jsonl", args.jsonl, "--model", args.model, "--out", args.out_model,
            "--epochs", "2", "--batch", "4", "--grad_accum", "4", "--lr", "2e-4", "--fp16"])

    if args.cmd in ("eval","all"):
        sh([sys.executable, P("eval_qa.py"),
            "--model", args.model, "--ft", args.out_model, "--out", "eval_raw_outputs.csv"])

if __name__ == "__main__":
    main()
