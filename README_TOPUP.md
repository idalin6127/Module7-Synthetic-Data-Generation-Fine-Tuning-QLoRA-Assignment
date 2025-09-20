
# Week7 Abstracts Top-up Toolkit

This folder contains small utilities to **get you from 50 → 100 abstracts** quickly by:
1) Enriching your Week4 `meta.jsonl` IDs with **abstracts** from the arXiv API.
2) Topping up **extra abstracts** from `cs.CL` / `cs.AI` / `cs.LG` while avoiding duplicates.
3) Merging results into a single **abstracts_100.json**.

## Files
- `enrich_with_abstracts.py` — read Week4 `meta.jsonl`, fetch abstracts for those IDs, save `abstracts.json`
- `top_up_abstracts.py` — query newest entries (e.g., `cs.CL cs.AI cs.LG`) to get N extra abstracts
- `merge_abstracts.py` — merge & dedupe two JSON files
- (Earlier created) `generate_jsonl_from_qas.py` — convert aggregated Q&A into instruction-tuning JSONL

## Recommended Workflow
> Run these commands in an environment with internet access (Colab/local).

### 1) Enrich Week4 IDs with abstracts
```
python enrich_with_abstracts.py   --meta /path/to/rag_arxiv_cscl_week4/data/meta.jsonl   --out abstracts.json
```

### 2) If abstracts.json has < 100 items, top up the rest
```
python top_up_abstracts.py   --existing abstracts.json   --cats cs.CL cs.AI cs.LG   --need 50   --out topup_abstracts.json
```

### 3) Merge to 100 (or more) and save
```
python merge_abstracts.py   --a abstracts.json   --b topup_abstracts.json   --out abstracts_100.json   --limit 100
```

### 4) Generate Q&A with GPT-4
For each item in `abstracts_100.json`, feed `abstract` into your prompt (see `qa_prompt_template.txt`)
to produce ~5 Q&A. Consolidate to `papers_qas.json` (aggregated format).

### 5) Convert Q&A → JSONL for training
```
python generate_jsonl_from_qas.py   --input papers_qas.json   --out synthetic_qa.jsonl
```

Then proceed to **QLoRA fine-tuning** using your notebook:
- `Week7_QLoRA_Finetune_AcademicQA.ipynb`

## Notes
- Be polite to arXiv API: ≤1 request/sec.
- Categories: start with `cs.CL`; to diversify, add `cs.AI` and `cs.LG` (or `stat.ML`). 
- Always **dedupe by arXiv ID** before counting to 100.
- If your Week4 bundle already downloaded many PDFs, step (1) will likely yield way more than 50 immediately.
