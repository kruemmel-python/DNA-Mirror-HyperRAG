
"""
dna_mirror_hyperrag.app
-----------------------
FastAPI-Server, der die Kernbibliothek lädt und Anfragen beantwortet.

Start (lokal):
  uvicorn dna_mirror_hyperrag.app:app --reload

Um Datenquellen zu steuern:
  - Umgebungsvariablen:
      RAG_MD_PATHS="sample_data/docs.md"
      RAG_JSON_PATHS="sample_data/kb.json"
"""

from __future__ import annotations

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dna_mirror_hyperrag.core import (
    RAGConfig, DNAMirrorHyperRAG, Neuromodulators, HyperGraph, Gene, HyperEdge
)
from dna_mirror_hyperrag.loaders import build_hgraph_from_sources


# -----------------------------
# Konfiguration laden
# -----------------------------

def _paths_from_env(var: str) -> list[str]:
    v = os.getenv(var, "").strip()
    return [p.strip() for p in v.split(",") if p.strip()] if v else []


MD_PATHS = _paths_from_env("RAG_MD_PATHS") or ["sample_data/docs.md"]
JSON_PATHS = _paths_from_env("RAG_JSON_PATHS") or ["sample_data/kb.json"]

hg: HyperGraph = build_hgraph_from_sources(md_files=MD_PATHS, json_files=JSON_PATHS,
                                           default_promoters=[("spiegelneuronen", 0.3), ("dna", 0.2)],
                                           json_regulatory={
                                               "neuro": [("imitationslernen", 0.2)],
                                               "dna": [("k-mer", 0.25), ("promoter", 0.15)],
                                               "sicherheit": [("risiken", 0.2)],
                                           })
rag = DNAMirrorHyperRAG(hg, RAGConfig(top_k=5, synthesis_max_sentences=4, kmer_k=3, default_view="grundlagen"))


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="DNA-Mirror-HyperRAG", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    dopamin: float = 1.0
    serotonin: float = 1.0
    gaba: float = 1.0


class QueryResponse(BaseModel):
    strategy: str
    weight: float
    rationale: str
    results: list[dict]
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "nodes": len(hg.nodes), "edges": len(hg.edges)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        neuromod = Neuromodulators(dopamin=req.dopamin, serotonin=req.serotonin, gaba=req.gaba)
        result = rag.answer(req.query, neuromod=neuromod)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Optional: minimale Endpunkte zum Debuggen der Graphstruktur

class AddGeneRequest(BaseModel):
    id: str
    sequence: str
    title: Optional[str] = None
    sites: Optional[List[tuple[str, str, float]]] = None  # [(type, pattern, boost)]
    metadata: Optional[dict] = None


@app.post("/debug/add_gene")
def add_gene(req: AddGeneRequest):
    if req.id in hg.nodes:
        raise HTTPException(status_code=400, detail=f"Gene-ID existiert bereits: {req.id}")
    sites = []
    if req.sites:
        for t, p, b in req.sites:
            from dna_mirror_hyperrag.core import RegulatorySite
            sites.append(RegulatorySite(t, p, float(b)))
    g = Gene(id=req.id, sequence=req.sequence, sites=sites, metadata=req.metadata or {"title": req.title or req.id})
    hg.add_gene(g)
    # Rebuild Index (einfachheitshalber)
    from dna_mirror_hyperrag.core import DNAIndex
    rag.index = DNAIndex(k=rag.config.kmer_k).build_from(hg.nodes.values())
    return {"status": "added", "id": req.id}


class AddEdgeRequest(BaseModel):
    id: str
    label: str
    members: List[str]


@app.post("/debug/add_edge")
def add_edge(req: AddEdgeRequest):
    if req.id in hg.edges:
        raise HTTPException(status_code=400, detail=f"Edge-ID existiert bereits: {req.id}")
    e = HyperEdge(id=req.id, label=req.label, members=set(req.members))
    hg.add_edge(e)
    return {"status": "added", "id": req.id}
