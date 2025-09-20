#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-paper Q&A from arXiv abstracts using OpenAI (GPT-4 class) and save to papers_qas.json.

Usage:
  export OPENAI_API_KEY=sk-...
  python generate_qas_from_abstracts.py --input abstracts_100.json --out papers_qas.json \
      --model gpt-4o --per_paper 5 --start 0 --end 100

Notes:
  - Input file format: a JSON list of objects, each containing at least:
      {"arxiv_id": "...", "title": "...", "authors": [...], "abstract": "...", "pdf_url": "..."}
  - Output file format: a JSON list of objects:
      [{"paper_id": "...", "title": "...", "qas": [{"question": "...", "answer": "..."}, ...]}, ...]
  - The script is sequential with retry & backoff. Adjust sleep if you hit rate limits.
"""

import os, sys, re, json, time, argparse, math, random
from typing import List, Dict

import re, json

def parse_json_list(text: str):
    """Try hard to parse a JSON array from possibly messy model output."""
    s = text.strip()

    # 1) 去掉 ```json / ``` 围栏
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 2) 直接尝试解析
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 3) 从整段文本里截取第一个 [ ... ] 片段再解析
    start = s.find("[")
    end   = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        frag = s[start:end+1]
        data = json.loads(frag)
        if isinstance(data, list):
            return data

    raise ValueError("Model did not return a JSON list. Raw content:\n" + text[:800])


# OpenAI SDK v1
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai>=1.0.0")
    sys.exit(1)

PROMPT_TEMPLATE = """You are a research assistant who reads academic papers and creates quiz questions.

Below is the abstract of a research paper. Read it and generate {per_paper} question–answer pairs that a student might ask.

STRICT OUTPUT RULES:
- Return ONLY a JSON array of EXACTLY {per_paper} objects. No prose, no explanations, no markdown fences, no backticks.
- Each object MUST have two keys: "question" and "answer".
- "question" <= 25 words. "answer" <= 60 words (2–4 sentences). Be concise.

CONTENT RULES:
- Cover key concepts, methods, results, limitations, comparisons.
- Answers must be based ONLY on the abstract (no outside knowledge).
- Include exactly ONE edge-case question whose answer must clearly say the abstract does not specify the asked detail, if it is not present.

Abstract:
\"\"\"{abstract_text}\"\"\""""


def ask_openai(client, model: str, prompt: str, max_retries: int = 5, timeout: int = 120) -> str:
    """Call OpenAI Chat Completions with basic retry + backoff. Returns the text content."""
    delay = 2.0
    for attempt in range(1, max_retries+1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("Empty content from API")
            return content
        except Exception as e:
            if attempt == max_retries:
                raise
            # exponential backoff
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 30.0)
    raise RuntimeError("Unreachable")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="abstracts JSON path (e.g., abstracts_100.json)")
    ap.add_argument("--out", required=True, help="Output path for papers_qas.json")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model name (e.g., gpt-4o, gpt-4.1, o4-mini)")
    ap.add_argument("--per_paper", type=int, default=5, help="Q&A pairs per paper")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive) for processing papers")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive) for processing papers; -1 means all")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between requests (be polite)")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Please export it before running.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    with open(args.input, "r", encoding="utf-8") as f:
        abstracts = json.load(f)

    n_total = len(abstracts)
    start = max(0, args.start)
    end = n_total if args.end < 0 else min(args.end, n_total)

    print(f"Loaded {n_total} abstracts. Processing range [{start}, {end}) with model={args.model}.")

    outputs = []
    for i in range(start, end):
        item = abstracts[i]
        arxiv_id = item.get("arxiv_id") or item.get("id") or f"paper_{i}"
        title = item.get("title", "").strip()
        abstract_text = item.get("abstract", "").strip()

        if not abstract_text:
            print(f"[{i}] {arxiv_id} has empty abstract. Skipping.")
            continue

        prompt = PROMPT_TEMPLATE.format(per_paper=args.per_paper, abstract_text=abstract_text)
        print(f"[{i}] Asking model for {arxiv_id} ...")
        try:
            content = ask_openai(client, args.model, prompt)
            # Expect JSON list
            qas = parse_json_list(content)
            
            cleaned = []
            for qa in qas:
                q = str(qa.get("question", "")).strip()
                a = str(qa.get("answer", "")).strip()
                if q and a:
                    cleaned.append({"question": q, "answer": a})
            if not cleaned:
                raise ValueError("No valid Q&A pairs parsed.")

            outputs.append({
                "paper_id": arxiv_id,
                "title": title,
                "qas": cleaned
            })
        except Exception as e:
            print(f"[{i}] Failed for {arxiv_id}: {e}")
        time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(outputs, w, ensure_ascii=False, indent=2)

    print(f"Saved {len(outputs)} papers with Q&A to {args.out}")

if __name__ == "__main__":
    main()
