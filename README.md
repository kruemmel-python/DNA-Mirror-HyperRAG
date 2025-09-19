
# DNA-Mirror-HyperRAG (Python 3.12)

Ein DNA-inspiriertes Retrieval-Augmented-Generation (RAG) System mit Hypergraph-Folding
und einer spiegel-neurologischen Entscheidungsschicht.

## Schnellstart

```bash
# (optional) Python venv aktivieren
pip install -r requirements.txt

# Tests
pytest -q

# Server starten
uvicorn dna_mirror_hyperrag.app:app --reload
# Dann: POST http://127.0.0.1:8000/query  {"query":"Erkläre DNA-RAG ..."}
```

## Umgebungsvariablen

- `RAG_MD_PATHS`  (kommasepariert), z. B. `sample_data/docs.md`
- `RAG_JSON_PATHS` (kommasepariert), z. B. `sample_data/kb.json`

## Architektur

- `core.py`   – Kernlogik: Gene/HyperGraph/DNAIndex/Mirror/RAG-Pipeline
- `loaders.py`– Loader für Markdown/JSON und Graph-Builder
- `app.py`    – FastAPI-Server (Health/Query + Debug-Endpunkte)
- `sample_data/` – Minimale Demo-Daten
- `tests/`    – pytest Smoke-Tests
