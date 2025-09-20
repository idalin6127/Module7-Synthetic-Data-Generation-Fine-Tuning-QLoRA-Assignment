#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top up abstracts to a target count by querying recent arXiv entries.
- Inputs:
    --existing abstracts.json (optional) to avoid duplicates
- Outputs:
    topup_abstracts.json  (new unique items)
Usage:
  python top_up_abstracts.py --out topup_abstracts.json --need 50 --cats cs.CL cs.AI cs.LG
Notes:
  - Uses arXiv API (Atom feed) with pagination via 'start' & 'max_results'.
  - Sorts by submitted date (newest first) and collects until 'need' unique are found.
"""
import argparse, json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

API = "http://export.arxiv.org/api/query"

def query_arxiv(categories, start=0, max_results=50, sortBy="submittedDate", sortOrder="descending"):
    q = " OR ".join([f"cat:{c}" for c in categories])
    params = {
        "search_query": q,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        xml = resp.read()
    return ET.fromstring(xml)

def parse_feed(root):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        arxiv_id = entry.findtext("a:id", default="", namespaces=ns).split("/")[-1]
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        authors = [a.findtext("a:name", default="", namespaces=ns).strip() for a in entry.findall("a:author", ns)]
        pdf_url = ""
        for ln in entry.findall("a:link", ns):
            if ln.attrib.get("type") == "application/pdf":
                pdf_url = ln.attrib.get("href", "")
                break
        out.append({"arxiv_id": arxiv_id, "title": title, "authors": authors, "abstract": summary, "pdf_url": pdf_url})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", type=str, default="", help="Existing abstracts JSON to avoid duplicates")
    ap.add_argument("--cats", nargs="+", default=["cs.CL"], help="arXiv categories")
    ap.add_argument("--need", type=int, default=50, help="How many new unique abstracts to collect")
    ap.add_argument("--out", type=str, default="topup_abstracts.json", help="Output JSON path")
    args = ap.parse_args()

    existing_ids = set()
    if args.existing:
        try:
            with open(args.existing, "r", encoding="utf-8") as f:
                for it in json.load(f):
                    existing_ids.add(it.get("arxiv_id", ""))
        except Exception:
            pass

    collected = []
    start = 0
    page = 0
    while len(collected) < args.need and page < 40:  # up to 40 pages * 50 = 2000 items
        root = query_arxiv(args.cats, start=start, max_results=50)
        entries = parse_feed(root)
        if not entries:
            break
        for e in entries:
            if e["arxiv_id"] and e["arxiv_id"] not in existing_ids:
                existing_ids.add(e["arxiv_id"])
                collected.append(e)
                if len(collected) >= args.need:
                    break
        page += 1
        start += 50
        time.sleep(1.0)
        print(f"Collected {len(collected)}/{args.need} so far...")
    with open(args.out, "w", encoding="utf-8") as w:
        json.dump(collected, w, ensure_ascii=False, indent=2)
    print(f"Saved {len(collected)} new abstracts to {args.out}")

if __name__ == "__main__":
    main()
