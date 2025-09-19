
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
    assert "quantum_jump_factor" in j


def test_upload_text_file_enables_querying():
    base_health = client.get("/health").json()
    unique_token = "sonderworthyperrag"
    text = f"Kapitel Eins\n\nDieses Buch enthält das Signalwort {unique_token} für den Test."

    files = {"files": ("buch.txt", text, "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["added_genes"]
    assert payload["total_nodes"] >= base_health["nodes"] + len(payload["added_genes"])

    query_resp = client.post("/query", json={"query": unique_token})
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert any(res["id"].startswith("TXT_buch_") for res in data["results"])
