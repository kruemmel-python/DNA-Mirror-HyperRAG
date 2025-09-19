
"""
Evaluation (nDCG@k, MAP) für DNA-Mirror-HyperRAG.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any
from dna_mirror_hyperrag.app import rag
from dna_mirror_hyperrag.core import Neuromodulators

def dcg(rels: List[int]) -> float:
    import math
    return sum(r / math.log2(i+2) for i, r in enumerate(rels))

def ndcg_at_k(pred: List[str], rel: List[str], k: int) -> float:
    relset = set(rel)
    gains = [1 if t in relset else 0 for t in pred[:k]]
    best = sorted(gains, reverse=True)
    return (dcg(gains) / dcg(best)) if dcg(best) > 0 else 0.0

def average_precision(pred: List[str], rel: List[str], k: int) -> float:
    relset = set(rel)
    hits = 0; precisions = []
    for i, t in enumerate(pred[:k], start=1):
        if t in relset:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / max(1, len(relset))

def evaluate(gt_path: str | Path, k: int = 5) -> Dict[str, Any]:
    ndcgs = []; maps = []
    for line in Path(gt_path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        item = json.loads(line)
        res = rag.answer(item["query"], neuromod=Neuromodulators())
        pred_titles = [r["title"] for r in res["results"]]
        ndcgs.append(ndcg_at_k(pred_titles, item["relevant_titles"], k))
        maps.append(average_precision(pred_titles, item["relevant_titles"], k))
    return {"n": len(ndcgs), "nDCG@k": sum(ndcgs)/max(1,len(ndcgs)), "MAP@k": sum(maps)/max(1,len(maps))}

if __name__ == "__main__":
    print(evaluate("sample_data/ground_truth.jsonl", k=5))
