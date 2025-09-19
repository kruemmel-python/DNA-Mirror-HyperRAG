
"""
dna_mirror_hyperrag.app
FastAPI-App, die den DNA-Mirror-HyperRAG-Dienst bereitstellt.
"""
from __future__ import annotations

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dna_mirror_hyperrag.core import (
    RAGConfig, DNAMirrorHyperRAG, Neuromodulators, HyperGraph, Gene, HyperEdge, DNAIndex
)
from dna_mirror_hyperrag.loaders import build_hgraph_from_sources

def _paths(var: str) -> list[str]:
    v = os.getenv(var, "").strip()
    return [p.strip() for p in v.split(",") if p.strip()] if v else []

MD_PATHS = _paths("RAG_MD_PATHS") or ["sample_data/docs.md"]
JSON_PATHS = _paths("RAG_JSON_PATHS") or ["sample_data/kb.json"]

hg: HyperGraph = build_hgraph_from_sources(
    md_files=MD_PATHS, json_files=JSON_PATHS,
    default_promoters=[("spiegelneuronen", 0.3), ("dna", 0.2)],
    json_regulatory={
        "neuro": [("imitationslernen", 0.2)],
        "dna": [("k-mer", 0.25), ("promoter", 0.15)],
        "sicherheit": [("risiken", 0.2)],
    }
)

rag = DNAMirrorHyperRAG(hg, RAGConfig(top_k=5, synthesis_max_sentences=4, kmer_k=3, default_view="grundlagen", dynamic_k=True))

app = FastAPI(title="DNA-Mirror-HyperRAG", version="0.3.0")

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
    return {"status": "ok", "nodes": len(hg.nodes), "edges": len(hg.edges), "k": rag.index.k}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        neuromod = Neuromodulators(req.dopamin, req.serotonin, req.gaba)
        result = rag.answer(req.query, neuromod=neuromod)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Debug: Gene hinzufügen
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
    from dna_mirror_hyperrag.core import RegulatorySite
    sites = [RegulatorySite(t,p,float(b)) for (t,p,b) in (req.sites or [])]
    g = Gene(req.id, req.sequence, sites=sites, metadata=req.metadata or {"title": req.title or req.id})
    hg.add_gene(g)
    # Index neu bauen (einfachheitshalber)
    rag.index = DNAIndex(k=rag.config.kmer_k).build_from(hg.nodes.values())
    return {"status":"added","id":req.id}

# Minimal-UI
@app.get("/ui")
def ui():
    html = """
<!doctype html>
<html><head><meta charset='utf-8'><title>DNA-Mirror-HyperRAG UI</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;max-width:900px;margin:40px auto;padding:0 16px;line-height:1.5}
.card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 2px 6px rgba(0,0,0,.05)}
code,pre{background:#f7f7f7;border-radius:8px;padding:4px 6px}
button{padding:8px 14px;border-radius:10px;border:1px solid #ccc;cursor:pointer}
input[type=range]{width:200px}
.small{color:#666;font-size:12px}
</style></head>
<body>
<h1>DNA-Mirror-HyperRAG</h1>
<p>Leichtgewichtige Demo-UI. Stelle eine Anfrage und passe Neuromodulatoren an.</p>
<div class="card">
  <label>Query<br><textarea id="q" rows="3" style="width:100%;">Erkläre DNA-basiertes RAG mit k-Mer Hybridisierung</textarea></label>
  <div style="display:flex;gap:16px;align-items:center;margin-top:10px;flex-wrap:wrap;">
    <div>Dopamin <input type="range" id="dop" min="0" max="2" step="0.1" value="1.0"><span id="dopv">1.0</span></div>
    <div>Serotonin <input type="range" id="ser" min="0" max="2" step="0.1" value="1.0"><span id="serv">1.0</span></div>
    <div>GABA <input type="range" id="gab" min="0" max="2" step="0.1" value="1.0"><span id="gabv">1.0</span></div>
    <button id="go">Anfragen</button>
  </div>
</div>
<div id="out"></div>
<p class="small">API: <code>POST /query</code> – Diese UI verwendet <code>fetch</code>.</p>
<script>
const $=s=>document.querySelector(s);
['dop','ser','gab'].forEach(id=>{
  const el=$( '#' + id ); const out=$( '#' + id + 'v' );
  el.addEventListener('input',()=>out.textContent=el.value);
});
$('#go').addEventListener('click', async ()=>{
  const body = {
    query: $('#q').value,
    dopamin: parseFloat($('#dop').value),
    serotonin: parseFloat($('#ser').value),
    gaba: parseFloat($('#gab').value)
  };
  const r = await fetch('/query', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  $('#out').innerHTML = `<div class="card">
    <div><b>Strategie:</b> ${j.strategy} | <b>Weight:</b> ${j.weight.toFixed(2)}</div>
    <div class="small">${j.rationale}</div>
    <pre>${j.answer.replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}</pre>
  </div>`;
});
</script>
</body></html>
"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(html)
