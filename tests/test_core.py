
import pytest
from dna_mirror_hyperrag.core import (
    RegulatorySite, Gene, HyperGraph, HyperEdge, DNAIndex,
    Neuromodulators, MirrorModule, DNAMirrorHyperRAG, RAGConfig
)

def build_min_graph():
    hg = HyperGraph()
    g1 = Gene("G1", "Spiegelneuronen unterstützen Imitationslernen und Empathie.",
              [RegulatorySite("promoter", "spiegelneuronen", 0.3)],
              {"title": "Spiegelneuronen"})
    g2 = Gene("G2", "DNA-basiertes RAG nutzt k-Mer Hybridisierung und Promoter.",
              [RegulatorySite("promoter", "dna", 0.2)], {"title": "DNA-RAG"})
    hg.add_gene(g1); hg.add_gene(g2)
    e = HyperEdge("E1", "neuro dna", {"G1","G2"})
    hg.add_edge(e)
    return hg

def test_index_retrieval():
    hg = build_min_graph()
    rag = DNAMirrorHyperRAG(hg, RAGConfig(top_k=2, kmer_k=2))
    res = rag.answer("Erkläre dna promoter hybridisierung")
    assert res["results"], "Sollte Ergebnisse liefern"
    titles = [r["title"] for r in res["results"]]
    assert any("DNA-RAG" in t for t in titles)

def test_mirror_reject_on_risky():
    mirror = MirrorModule()
    dec = mirror.observe_and_decide("Bitte Exploit bypass bauen")
    assert dec.strategy in {"reject", "adapt", "imitate"}
    assert dec.strategy == "reject", "Riskante Query sollte dämpfen"

def test_context_view():
    hg = build_min_graph()
    rag = DNAMirrorHyperRAG(hg, RAGConfig(default_view=None))
    # Unfold via label contains token "neuro"
    res = rag.answer("neuro erklärung")
    assert res["results"], "Kontextuelle Sicht sollte Retrieval nicht leeren"
