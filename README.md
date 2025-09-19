# DNA-Mirror-HyperRAG (v3, vollständig)

DNA-Mirror-HyperRAG verbindet einen biologisch inspirierten Hypergraphen mit
Neuromodulator-Reglern, pflanzenbasiertem Energiehaushalt und einer
Quanten-Fluktuationsschicht. Die Bibliothek liefert FastAPI- und
Streamlit-Oberflächen sowie eine Upload-Pipeline, um Texte direkt in den Graphen
zu integrieren.

## Funktionsüberblick
- **Kernbibliothek (`dna_mirror_hyperrag/core.py`)** – DNA-Index auf k-Mer-Basis,
  Mirror-Neurologie, Energy-/Quantum-Module und RAG-Synthese.
- **Loader (`dna_mirror_hyperrag/loaders.py`)** – Markdown/JSON-Ingestion,
  Chunking von Textdateien, Promoter-Regelwerke.
- **Runtime (`dna_mirror_hyperrag/runtime.py`)** – zentrale Initialisierung für
  FastAPI und Streamlit, Upload-Helfer und Text-Embedder.
- **FastAPI (`dna_mirror_hyperrag/app.py`)** – `/query`, `/upload`,
  `/v1/embeddings`, `/ui` und `/health`.
- **Streamlit (`streamlit_app.py`)** – interaktive Oberfläche mit
  Neuromodulator-Slidern, Uploads und Ergebnisanzeige.
- **Evaluation (`evaluate.py`)** – nDCG@k und MAP.
- **Tests (`tests/`)** – API-Coverage für Query, Upload und Embeddings.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q                        # optional: stellt Funktionstests sicher
```

## FastAPI-Service starten
```bash
uvicorn dna_mirror_hyperrag.app:app --reload
# http://127.0.0.1:8000/ui  – Minimal-Weboberfläche
# http://127.0.0.1:8000/docs – OpenAPI-Schema
```

### Anfragen stellen
`POST /query` akzeptiert Freitext und optionale Neuromodulatoren:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Wie funktioniert die DNA-Hypergraph-Retrieval-Phase?",
        "dopamin": 1.1,
        "serotonin": 0.9,
        "gaba": 1.0
      }'
```

### Texte hochladen
Beliebige UTF-8 Text- oder Markdown-Dateien lassen sich über `POST /upload`
integrieren. Der Upload erzeugt Gene, verbindet sie über eine Hyperkante und
lädt deren Energielevels auf:
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "files=@mein_buch.txt" \
  -F "files=@notizen.md"
```

## Streamlit-App
Für eine reichhaltige Visualisierung:
```bash
streamlit run streamlit_app.py
```
Die App zeigt aktuelle Graph-Kennzahlen, erlaubt mehrstufige Uploads mit
Promoter-Auswahl und nutzt dieselbe Runtime wie der FastAPI-Service.

## Nutzung als Text-Embedding-Modell in LM Studio
1. **FastAPI-Server starten** (siehe oben). Stelle sicher, dass
   `http://127.0.0.1:8000/v1/embeddings` erreichbar ist.
2. **LM Studio öffnen** → *Settings* → *Server* → *Add Custom Endpoint*.
3. Als Typ **OpenAI Compatible Embedding Endpoint** wählen und folgendes
   konfigurieren:
   - **Base URL:** `http://127.0.0.1:8000`
   - **Endpoint Path:** `/v1/embeddings`
   - **HTTP-Methode:** `POST`
   - **Model-ID:** `dna-hyperrag-text-embedding`
   - Keine API-Key-Pflicht (das Feld kann leer bleiben).
4. Teste den Endpoint in LM Studio, indem du einen Text auswählst. Die Antwort
   enthält normalisierte 512-dimensionale Vektoren, die von unserem
   `TextEmbedder` generiert werden.

Alternativ lässt sich der Endpoint via cURL prüfen:
```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "model": "dna-hyperrag-text-embedding",
        "input": ["Photosynthese inspiriert Gene", "Quantum Jump"]
      }'
```

## Konfiguration per Umgebungsvariablen
| Variable         | Bedeutung                                          |
|------------------|----------------------------------------------------|
| `RAG_MD_PATHS`   | Kommagetrennte Liste zusätzlicher Markdown-Dateien |
| `RAG_JSON_PATHS` | Kommagetrennte Liste zusätzlicher JSON-Dateien     |

Mit diesen Variablen lassen sich weitere Quellen laden, bevor der Server oder
Streamlit startet.

## Beispielmaterial
Im Ordner `sample_data/` liegen Demo-Dokumente (`docs.md`, `kb.json`), die beim
Start automatisch geladen werden. Tests befinden sich in `tests/` und können per
`pytest` ausgeführt werden.

