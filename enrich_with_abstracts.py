#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich a list of arXiv IDs (from Week4 meta.jsonl) with abstracts using the public arXiv API.
- Input: meta.jsonl with lines like {"id": "http://arxiv.org/abs/2508.10880v1", "title": "...", ...}
- Output: abstracts.json with [{"arxiv_id": "2508.10880v1", "title": "...", "authors": [...], "abstract": "...", "pdf_url": "..."}]
Usage:
  python enrich_with_abstracts.py --meta path/to/meta.jsonl --out abstracts.json
Notes:
  - Respects arXiv API rate limits (<= 1 req/sec). 
  - Requires internet access to run.
"""
import argparse, json, re, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"

def parse_id(url_or_id: str) -> str:
    # Accept 'http://arxiv.org/abs/2508.10880v1' or '2508.10880v1' and return bare id
    m = re.search(r'(\d{4}\.\d{4,5}(v\d+)?)', url_or_id)
    if m:
        return m.group(1)
    return url_or_id.strip()

def fetch_entry(arxiv_id: str) -> dict:
    q = f"id_list={urllib.parse.quote(arxiv_id)}"
    url = f"{API}?{q}"
    with urllib.request.urlopen(url) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return {}
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    authors = [a.findtext("a:name", default="", namespaces=ns).strip() for a in entry.findall("a:author", ns)]
    links = entry.findall("a:link", ns)
    pdf_url = ""
    for ln in links:
        if ln.attrib.get("type") == "application/pdf":
            pdf_url = ln.attrib.get("href", "")
            break
    return {"arxiv_id": arxiv_id, "title": title, "authors": authors, "abstract": summary, "pdf_url": pdf_url}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to meta.jsonl from Week4")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of IDs to fetch")
    args = ap.parse_args()

    ids = []
    with open(args.meta, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ids.append(parse_id(obj.get("id", "")))
    # Dedup & keep order
    seen = set()
    ordered_ids = []
    for aid in ids:
        if aid and aid not in seen:
            seen.add(aid)
            ordered_ids.append(aid)

    if args.limit and args.limit > 0:
        ordered_ids = ordered_ids[:args.limit]

    results = []
    for i, aid in enumerate(ordered_ids, 1):
        data = fetch_entry(aid)
        if data:
            results.append(data)
        print(f"[{i}/{len(ordered_ids)}] fetched {aid} -> {'OK' if data else 'MISS'}")
        time.sleep(1.0)  # be nice to API

    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(results, w, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} items to {args.out}")

if __name__ == "__main__":
    main()
