
"""FastAPI surface for the DNA-Mirror-HyperRAG service."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from dna_mirror_hyperrag.core import Neuromodulators
from dna_mirror_hyperrag.runtime import (
    RuntimeState,
    initialize_runtime,
    ingest_text_document,
)


runtime: RuntimeState = initialize_runtime()
hg = runtime.graph
rag = runtime.rag

app = FastAPI(title="DNA-Mirror-HyperRAG", version="0.4.0")

class QueryRequest(BaseModel):
    query: str
    dopamin: float = 1.0
    serotonin: float = 1.0
    gaba: float = 1.0

class QueryResponse(BaseModel):
    strategy: str
    weight: float
    rationale: str
    quantum_jump_factor: float
    results: list[dict]
    answer: str


class UploadResponse(BaseModel):
    added_genes: list[str]
    added_edges: list[str]
    total_nodes: int
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
    from dna_mirror_hyperrag.core import RegulatorySite, Gene, DNAIndex

    sites = [RegulatorySite(t, p, float(b)) for (t, p, b) in (req.sites or [])]
    gene = Gene(
        req.id,
        req.sequence,
        sites=sites,
        metadata=req.metadata or {"title": req.title or req.id},
    )
    hg.add_gene(gene)
    rag.index = DNAIndex(k=rag.config.kmer_k).build_from(
        hg.nodes.values(), energy_module=rag.light_energy_module
    )
    return {"status": "added", "id": req.id}


@app.post("/upload", response_model=UploadResponse)
async def upload_text_files(
    files: List[UploadFile] = File(...),
    tokens_per_chunk: int = 220,
    overlap: int = 40,
):
    if not files:
        raise HTTPException(status_code=400, detail="Keine Dateien übermittelt.")

    added_genes: list[str] = []
    added_edges: list[str] = []

    for file in files:
        raw = await file.read()
        if not raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"Datei {file.filename!r} ist keine UTF-8 Textdatei.",
            )

        result = ingest_text_document(
            text,
            file.filename or "upload",
            runtime,
            tokens_per_chunk=tokens_per_chunk,
            overlap=overlap,
        )
        added_genes.extend(result["added_genes"])
        if result["edge_id"]:
            added_edges.append(result["edge_id"])

    if not added_genes:
        raise HTTPException(status_code=400, detail="Keine gültigen Textdaten gefunden.")

    return UploadResponse(
        added_genes=added_genes,
        added_edges=added_edges,
        total_nodes=len(hg.nodes),
    )

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
    return HTMLResponse(html)
