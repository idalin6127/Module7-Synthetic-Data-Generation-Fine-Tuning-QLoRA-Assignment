#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two abstract JSON files, deduplicate by arXiv ID, and optionally truncate to a target count.
Usage:
  python merge_abstracts.py --a abstracts.json --b topup_abstracts.json --out abstracts_100.json --limit 100
"""
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="First JSON (e.g., enriched from Week4)")
    ap.add_argument("--b", required=True, help="Second JSON (e.g., top-up new abstracts)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0, help="Optional max size")
    args = ap.parse_args()

    items = []
    for p in [args.a, args.b]:
        with open(p, "r", encoding="utf-8") as f:
            items.extend(json.load(f))
    seen = set()
    merged = []
    for it in items:
        aid = it.get("arxiv_id", "")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        merged.append(it)
    if args.limit and len(merged) > args.limit:
        merged = merged[:args.limit]
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(merged, w, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} items -> {args.out}")

if __name__ == "__main__":
    main()
