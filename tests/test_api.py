
from fastapi.testclient import TestClient
from dna_mirror_hyperrag.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert "nodes" in j and "edges" in j

def test_query():
    r = client.post("/query", json={"query":"Wie funktioniert k-Mer Hybridisierung im DNA-RAG?"})
    assert r.status_code == 200
    j = r.json()
    assert "strategy" in j and "results" in j
