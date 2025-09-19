# DNA-Mirror-HyperRAG (v3, vollständig)

**Was drin ist**
- Vollständige **Kernbibliothek** (`core.py`) – DNA-Index (k-Mer), Hypergraph, Mirror-Neurologie, RAG-Pipeline
- **Loader** (`loaders.py`) für Markdown/JSON + Graph-Builder
- **FastAPI-App** (`app.py`) mit `/health`, `/query`, `/ui`
- **Evaluation** (`evaluate.py`) mit nDCG@k und MAP
- **Sample-Daten** (`sample_data/`) und **Tests** (`tests/`)

## Schnellstart (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
uvicorn dna_mirror_hyperrag.app:app --reload
# Browser: http://127.0.0.1:8000/ui
```

## Env-Variablen (optional)
- `RAG_MD_PATHS`  (kommaseparierte Pfade zu .md)
- `RAG_JSON_PATHS` (kommaseparierte Pfade zu .json)
