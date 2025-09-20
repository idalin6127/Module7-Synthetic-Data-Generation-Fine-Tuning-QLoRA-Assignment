#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate synthetic_qa.jsonl from per-paper Q&A JSON files or a single aggregated JSON.
# Usage examples:
#   python generate_jsonl_from_qas.py --input papers_qas.json --out synthetic_qa.jsonl
#   python generate_jsonl_from_qas.py --dir ./qas_per_paper --out synthetic_qa.jsonl
# Input formats:
#   1) Aggregated: [{"paper_id": "...", "title": "...", "qas": [{"question": "...", "answer": "..."}, ...]}, ...]
#   2) Per-file: Each file contains {"paper_id": "...", "title": "...", "qas": [...]}

import os, json, argparse, glob

SYSTEM_PROMPT = "You are a helpful academic Q&A assistant specialized in scholarly content."

def load_items(args):
    items = []
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            items = json.load(f)
    elif args.dir:
        for path in glob.glob(os.path.join(args.dir, "*.json")):
            with open(path, "r", encoding="utf-8") as f:
                items.append(json.load(f))
    else:
        raise ValueError("Provide --input or --dir")
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="Aggregated JSON file")
    ap.add_argument("--dir", type=str, help="Directory of per-paper JSON files")
    ap.add_argument("--out", type=str, default="synthetic_qa.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    items = load_items(args)
    n_pairs = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for it in items:
            qas = it.get("qas", [])
            for qa in qas:
                q = qa["question"].strip()
                a = qa["answer"].strip()
                full_prompt = f"<|system|>{SYSTEM_PROMPT}<|user|>{q}<|assistant|>{a}"
                w.write(json.dumps({"text": full_prompt}, ensure_ascii=False) + "\n")
                n_pairs += 1
    print(f"Wrote {n_pairs} pairs to {args.out}")

if __name__ == "__main__":
    main()
