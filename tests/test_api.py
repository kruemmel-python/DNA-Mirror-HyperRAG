
import io

import pytest
from fastapi import HTTPException, UploadFile

from dna_mirror_hyperrag.app import (
    EmbeddingRequest,
    embedder,
    create_embeddings,
    QueryRequest,
    UploadResponse,
    health,
    query,
    upload_text_files,
)



@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_health():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["nodes"] >= 1
    assert payload["edges"] >= 0


def test_query():
    response = query(QueryRequest(query="Wie funktioniert k-Mer Hybridisierung im DNA-RAG?"))
    assert "strategy" in response and "results" in response
    assert "quantum_jump_factor" in response


@pytest.mark.anyio
async def test_upload_text_file_enables_querying():
    base_health = health()
    unique_token = "sonderworthyperrag"
    text = (
        "Kapitel Eins\n\nDieses Buch enthält das Signalwort "
        f"{unique_token} für den Test."
    )

    upload = UploadFile(filename="buch.txt", file=io.BytesIO(text.encode("utf-8")))
    resp: UploadResponse = await upload_text_files(files=[upload])
    assert resp.added_genes
    assert resp.total_nodes >= base_health["nodes"] + len(resp.added_genes)

    data = query(QueryRequest(query=unique_token))
    assert any(res["id"].startswith("TXT_buch_") for res in data["results"])


def test_embeddings_endpoint_returns_normalised_vectors():
    response = create_embeddings(EmbeddingRequest(input=["Hallo Welt", "Hallo Welt"]))
    assert response.model == "dna-hyperrag-text-embedding"
    assert len(response.data) == 2
    assert response.data[0].embedding == response.data[1].embedding

    for item in response.data:
        assert len(item.embedding) == embedder.dimension
        norm = sum(v * v for v in item.embedding) ** 0.5
        assert norm == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_embeddings_endpoint_rejects_empty_payload():
    with pytest.raises(HTTPException):
        create_embeddings(EmbeddingRequest(input="   "))
