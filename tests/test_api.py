
import io

import pytest
from fastapi import UploadFile

from dna_mirror_hyperrag.app import (
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
