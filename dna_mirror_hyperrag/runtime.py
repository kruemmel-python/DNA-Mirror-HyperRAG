"""Runtime utilities shared between service frontends.

This module centralises the configuration and lifecycle helpers that were
previously embedded inside the FastAPI surface so that alternative entry
points – such as the Streamlit UI – can reuse the same behaviour without
duplicating logic.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dna_mirror_hyperrag.core import (
    DNAMirrorHyperRAG,
    Gene,
    HyperEdge,
    HyperGraph,
    RAGConfig,
)
from dna_mirror_hyperrag.embeddings import TextEmbedder
from dna_mirror_hyperrag.loaders import build_hgraph_from_sources, genes_from_text_document

__all__ = [
    "DEFAULT_PROMOTERS",
    "JSON_REGULATORY",
    "RuntimeState",
    "build_default_hgraph",
    "default_config",
    "initialize_runtime",
    "ingest_text_document",
    "sanitize_identifier",
]


DEFAULT_PROMOTERS = [("spiegelneuronen", 0.3), ("dna", 0.2)]

JSON_REGULATORY = {
    "neuro": [("imitationslernen", 0.2)],
    "dna": [("k-mer", 0.25), ("promoter", 0.15)],
    "sicherheit": [("risiken", 0.2)],
}


def _paths_from_env(var: str, fallback: Iterable[str]) -> list[str]:
    value = os.getenv(var, "").strip()
    if not value:
        return list(fallback)
    return [p.strip() for p in value.split(",") if p.strip()]


def sanitize_identifier(name: str) -> str:
    """Return a filesystem- and id-safe representation of *name*."""

    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return safe or "upload"


def build_default_hgraph() -> HyperGraph:
    """Construct the default hyper graph from environment configured sources."""

    md_paths = _paths_from_env("RAG_MD_PATHS", ["sample_data/docs.md"])
    json_paths = _paths_from_env("RAG_JSON_PATHS", ["sample_data/kb.json"])
    return build_hgraph_from_sources(
        md_files=md_paths,
        json_files=json_paths,
        default_promoters=DEFAULT_PROMOTERS,
        json_regulatory=JSON_REGULATORY,
    )


def default_config() -> RAGConfig:
    """Return the opinionated default configuration used by both frontends."""

    return RAGConfig(
        top_k=5,
        synthesis_max_sentences=4,
        kmer_k=3,
        default_view="grundlagen",
        dynamic_k=True,
        energy_decay_rate=0.005,
        energy_recharge_amount=0.1,
        min_gene_energy=0.1,
        quantum_jump_threshold=0.7,
        quantum_plasma_influence=0.5,
        quantum_max_jump_factor=0.3,
    )


@dataclass
class RuntimeState:
    """Mutable runtime container shared by FastAPI and Streamlit entrypoints."""

    graph: HyperGraph
    rag: DNAMirrorHyperRAG
    embedder: TextEmbedder


def initialize_runtime(config: RAGConfig | None = None) -> RuntimeState:
    """Initialise the hyper graph and retrieval engine using *config*."""

    cfg = config or default_config()
    graph = build_default_hgraph()
    rag = DNAMirrorHyperRAG(graph, cfg)
    embedder = TextEmbedder()
    return RuntimeState(graph=graph, rag=rag, embedder=embedder)


def _ensure_unique_gene_id(gene: Gene, graph: HyperGraph) -> Gene:
    base_id = gene.id
    suffix = 1
    while gene.id in graph.nodes:
        gene.id = f"{base_id}_{suffix}"
        suffix += 1
    return gene


def _ensure_unique_edge_id(edge_base: str, graph: HyperGraph) -> str:
    edge_id = edge_base
    suffix = 1
    while edge_id in graph.edges:
        edge_id = f"{edge_base}_{suffix}"
        suffix += 1
    return edge_id


def ingest_text_document(
    text: str,
    filename: str,
    state: RuntimeState,
    *,
    tokens_per_chunk: int = 220,
    overlap: int = 40,
    promoters: Iterable[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Chunk *text* into genes and integrate them into the runtime state.

    The function mirrors the previous FastAPI upload logic so that both the API
    and the Streamlit UI can reuse the same ingestion pathway.
    """

    graph, rag = state.graph, state.rag
    safe_base = sanitize_identifier(Path(filename or "upload").stem or "upload")
    genes = genes_from_text_document(
        text,
        safe_base,
        tokens_per_chunk=max(50, tokens_per_chunk),
        overlap=max(0, overlap),
        default_promoters=list(promoters or DEFAULT_PROMOTERS),
    )

    if not genes:
        return {"added_genes": [], "edge_id": None}

    added: list[str] = []
    for gene in genes:
        gene = _ensure_unique_gene_id(gene, graph)
        gene.metadata = dict(gene.metadata)
        gene.metadata.update({
            "original_filename": filename,
            "source": f"upload:{filename or safe_base}",
        })
        graph.add_gene(gene)
        rag.index.add(gene)
        added.append(gene.id)

    members = set(added)
    if members:
        edge_base = f"TXT_EDGE_{safe_base}"
        edge_id = _ensure_unique_edge_id(edge_base, graph)
        edge = HyperEdge(edge_id, filename or safe_base, members)
        graph.add_edge(edge)
        if "grundlagen" in graph.views:
            graph.views["grundlagen"].update(members)
    else:
        edge_id = None

    return {"added_genes": added, "edge_id": edge_id}

