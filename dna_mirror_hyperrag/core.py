
"""
dna_mirror_hyperrag.core
------------------------
Kernbibliothek für ein DNA-inspiriertes RAG mit spiegel-neurologischer Entscheidungslogik
und Hypergraph-Speicher (Folding/Unfolding). Python 3.12, ohne externe Abhängigkeiten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, Callable
from collections import defaultdict, Counter
import math


# -----------------------------
# Hilfsfunktionen
# -----------------------------

def _normalize_whitespace(s: str) -> str:
    """Trimmt Whitespaces auf Einzelspatien. Klar und deterministisch."""
    return " ".join(s.split())


def _tokenize(s: str) -> list[str]:
    """Einfacher Tokenizer: lowercase + Split auf Leerraum (didaktisch)."""
    return _normalize_whitespace(s).lower().split()


def _kmerize(tokens: Sequence[str], k: int) -> list[tuple[str, ...]]:
    """Erzeuge k-Mers (Token-n-Gramme). k>0 ist Pflicht – klare Fehlermeldung sonst."""
    if k <= 0:
        raise ValueError(f"k muss > 0 sein, erhalten: {k}")
    return [tuple(tokens[i:i + k]) for i in range(0, max(0, len(tokens) - k + 1))]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    """Einfacher BoW-Cosine (globales Maß) – stabilisiert lokale k-Mer-Ähnlichkeit."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# -----------------------------
# DNA-Regulatorik + Gene
# -----------------------------

@dataclass(frozen=True)
class RegulatorySite:
    """
    DNA-analoge Regulationsstelle:
    - type:  'promoter' | 'enhancer' | 'silencer'
    - pattern: Textmotiv für Query-Matching
    - boost:  Relevanzmodulator (>=0 sinnvoll)
    """
    type: str
    pattern: str
    boost: float = 0.0


@dataclass
class Gene:
    """
    Ein semantischer Wissens-Chunk.
    - sequence: Volltext (Absatz/Fakt)
    - sites:    Regulatorische Elemente
    - metadata: Metadaten (Titel, Tags, Quelle, etc.)
    """
    id: str
    sequence: str
    sites: list[RegulatorySite] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def tokens(self) -> list[str]:
        return _tokenize(self.sequence)

    def bow(self) -> Counter[str]:
        return Counter(self.tokens())


# -----------------------------
# Hypergraph
# -----------------------------

@dataclass
class HyperEdge:
    """
    Hyperkante verknüpft mehrere Gene über ein gemeinsames Motiv/Topic.
    - folded=True: standardmäßig „eingeklappt“ (inaktiv), bis Kontext entfaltet.
    """
    id: str
    label: str
    members: set[str]
    folded: bool = False


@dataclass
class HyperGraph:
    """
    Hypergraph als Wissensorganisation.
    - nodes: Gene
    - edges: Hyperkanten
    - views: benannte Sichten (Menge von Gene-IDs)
    """
    nodes: dict[str, Gene] = field(default_factory=dict)
    edges: dict[str, HyperEdge] = field(default_factory=dict)
    views: dict[str, set[str]] = field(default_factory=dict)

    def add_gene(self, gene: Gene) -> None:
        if gene.id in self.nodes:
            raise ValueError(f"Gene-ID bereits vorhanden: {gene.id!r}")
        self.nodes[gene.id] = gene

    def add_edge(self, edge: HyperEdge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"Edge-ID bereits vorhanden: {edge.id!r}")
        missing = [gid for gid in edge.members if gid not in self.nodes]
        if missing:
            raise ValueError(f"HyperEdge {edge.id!r} referenziert unbekannte Gene: {missing}")
        self.edges[edge.id] = edge

    def fold(self, edge_id: str) -> None:
        edge = self.edges.get(edge_id)
        if not edge:
            raise KeyError(f"Unbekannte HyperEdge: {edge_id!r}")
        edge.folded = True

    def unfold(self, edge_id: str) -> None:
        edge = self.edges.get(edge_id)
        if not edge:
            raise KeyError(f"Unbekannte HyperEdge: {edge_id!r}")
        edge.folded = False

    def unfold_by_label_contains(self, pattern: str) -> set[str]:
        """
        Entfalte alle Hyperkanten, deren Label den Pattern-String enthält.
        Liefert Menge der sichtbaren Gene.
        """
        pattern_l = pattern.lower()
        visible: set[str] = set()
        for e in self.edges.values():
            if pattern_l in e.label.lower():
                e.folded = False
                visible |= e.members
        return visible

    def make_view(self, name: str, visible_gene_ids: set[str]) -> None:
        missing = [gid for gid in visible_gene_ids if gid not in self.nodes]
        if missing:
            raise ValueError(f"View {name!r} enthält unbekannte Gene: {missing}")
        self.views[name] = set(visible_gene_ids)

    def get_view(self, name: str) -> set[str]:
        if name not in self.views:
            raise KeyError(f"Unbekannte View: {name!r}")
        return set(self.views[name])


# -----------------------------
# DNA-Index
# -----------------------------

@dataclass
class DNAIndex:
    """Invertierter k-Mer-Index über Gene (Hybridisierung)."""
    k: int = 3
    postings: dict[tuple[str, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    genes: dict[str, Gene] = field(default_factory=dict)

    def add(self, gene: Gene) -> None:
        if gene.id in self.genes:
            raise ValueError(f"Gene-ID bereits vorhanden: {gene.id!r}")
        self.genes[gene.id] = gene
        for kmer in _kmerize(gene.tokens(), self.k):
            self.postings[kmer].add(gene.id)

    def build_from(self, genes: Iterable[Gene]) -> "DNAIndex":
        for g in genes:
            self.add(g)
        return self

    def _regulatory_boost(self, gene: Gene, query: str) -> float:
        q = _normalize_whitespace(query).lower()
        boost = 0.0
        for site in gene.sites:
            pat = site.pattern.lower()
            if pat and pat in q:
                match site.type.lower():
                    case "promoter":
                        boost += abs(site.boost) + 0.2
                    case "enhancer":
                        boost += abs(site.boost)
                    case "silencer":
                        boost -= abs(site.boost)
                    case _:
                        boost += 0.0
        return boost

    def _hybridization(self, q_tokens: list[str], gene: Gene) -> float:
        q_kmers = _kmerize(q_tokens, self.k)
        if not q_kmers:
            return 0.0
        g_kmers = set(_kmerize(gene.tokens(), self.k))
        matches = sum(1 for km in q_kmers if km in g_kmers)
        return matches / len(q_kmers)

    def _bow_bonus(self, q_tokens: list[str], gene: Gene) -> float:
        return 0.15 * _cosine(Counter(q_tokens), gene.bow())

    def retrieve(self, query: str, limit: int = 5, restrict_to_ids: set[str] | None = None) -> list[tuple[Gene, float]]:
        if not query or not query.strip():
            raise ValueError("Leere Anfrage kann nicht abgerufen werden.")
        q_tokens = _tokenize(query)

        candidate_ids: set[str] = set()
        for km in _kmerize(q_tokens, self.k):
            candidate_ids |= self.postings.get(km, set())

        if not candidate_ids:
            candidate_ids = set(self.genes.keys())

        if restrict_to_ids is not None:
            candidate_ids &= restrict_to_ids
            if not candidate_ids:
                candidate_ids = restrict_to_ids

        scored: list[tuple[Gene, float]] = []
        for gid in candidate_ids:
            gene = self.genes[gid]
            base = self._hybridization(q_tokens, gene)
            reg = self._regulatory_boost(gene, query)
            bonus = self._bow_bonus(q_tokens, gene)
            score = max(0.0, base + reg + bonus)
            scored.append((gene, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# -----------------------------
# Spiegel-Neurologie
# -----------------------------

@dataclass
class Neuromodulators:
    """Dopamin (Anreiz), Serotonin (Kontrolle), GABA (Hemmung). Alle Regler 0..2."""
    dopamin: float = 1.0
    serotonin: float = 1.0
    gaba: float = 1.0

    def clamp(self) -> "Neuromodulators":
        def c(x: float) -> float:
            return max(0.0, min(2.0, x))
        return Neuromodulators(c(self.dopamin), c(self.serotonin), c(self.gaba))


@dataclass
class MirrorDecision:
    strategy: str   # "imitate" | "adapt" | "reject"
    weight: float
    rationale: str


class MirrorModule:
    """
    Beobachtet Query, bewertet (novelty, risk, clarity) und wählt Strategie.
    Ersetzt oder erweitert später leicht durch ein echtes Klassifikationsmodell.
    """
    def __init__(self, policy: Callable[[str], dict[str, float]] | None = None):
        self.policy = policy or self._default_policy

    def _default_policy(self, query: str) -> dict[str, float]:
        q = _normalize_whitespace(query).lower()
        toks = _tokenize(q)
        novelty = min(1.0, len(set(toks)) / 12)
        risk = 0.4 if any(w in q for w in ("hack", "exploit", "bypass", "waffe", "angreifen", "malware")) else 0.1
        clarity = 0.7 if len(q) >= 20 else 0.4
        return {"novelty": novelty, "risk": risk, "clarity": clarity}

    def observe_and_decide(self, query: str, neuromod: Neuromodulators = Neuromodulators()) -> MirrorDecision:
        if not query or not query.strip():
            raise ValueError("MirrorModule: Leere Anfrage kann nicht beobachtet werden.")
        feats = self.policy(query)
        neuromod = neuromod.clamp()

        imitate_drive = 0.6 * feats["clarity"] + 0.3 * (1 - feats["novelty"]) + 0.1 * neuromod.dopamin
        adapt_drive   = 0.5 * feats["novelty"] + 0.3 * feats["clarity"] + 0.2 * neuromod.serotonin
        reject_drive  = 0.6 * feats["risk"] + 0.2 * (1 - feats["clarity"]) + 0.2 * neuromod.gaba

        match max((("imitate", imitate_drive), ("adapt", adapt_drive), ("reject", reject_drive)), key=lambda x: x[1])[0]:
            case "imitate":
                weight = 1.0 + 0.5 * neuromod.dopamin
                rationale = "Klares bekanntes Muster → Imitation naheliegend."
                return MirrorDecision("imitate", weight, rationale)
            case "adapt":
                weight = 1.0 + 0.2 * neuromod.serotonin
                rationale = "Teilweise neuartig → vorsichtig transformieren."
                return MirrorDecision("adapt", weight, rationale)
            case "reject":
                weight = 0.5
                rationale = "Riskant/unklar → dämpfen und einschränken."
                return MirrorDecision("reject", weight, rationale)
            case _:
                raise RuntimeError("Unbekannte Mirror-Strategie – sollte nie auftreten.")


# -----------------------------
# RAG-Pipeline
# -----------------------------

@dataclass
class RAGConfig:
    top_k: int = 5
    synthesis_max_sentences: int = 4
    kmer_k: int = 3
    default_view: str | None = None  # Name einer vorkonfigurierten View (optional)


class DNAMirrorHyperRAG:
    """
    End-to-end Pipeline:
      1) Mirror-Entscheidung
      2) Hypergraph-Kontext -> Sicht
      3) DNA-Retrieval (ggf. auf Sicht)
      4) Score-Modulation + Strategie-spezifische Nachlogik
      5) Extraktive Synthesis (debuggbar)
    """
    def __init__(self, hgraph: HyperGraph, config: RAGConfig = RAGConfig()):
        self.hgraph = hgraph
        self.config = config
        self.index = DNAIndex(k=config.kmer_k).build_from(hgraph.nodes.values())
        self.mirror = MirrorModule()

    @staticmethod
    def _key_sentences(text: str, max_n: int) -> list[str]:
        raw = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
        sents = [s for s in raw if s]
        sents.sort(key=len, reverse=True)
        return sents[:max_n]

    def _synthesize(self, query: str, ranked: list[tuple[Gene, float]]) -> str:
        lines: list[str] = []
        lines.append(f"▶ Anfrage: {query}")
        lines.append("▶ Quellen (Top):")
        for g, score in ranked[: self.config.top_k]:
            title = g.metadata.get("title") or g.id
            lines.append(f"  - {title}  (Score: {score:.3f})")
        corpus = " ".join(g.sequence for g, _ in ranked[: self.config.top_k])
        bullets = self._key_sentences(corpus, self.config.synthesis_max_sentences)
        if bullets:
            lines.append("▶ Kernaussagen:")
            for b in bullets:
                lines.append(f"  • {b.strip()}.")
        return "\n".join(lines)

    def _context_to_view(self, query: str) -> set[str] | None:
        toks = _tokenize(query)
        visible: set[str] = set()
        for t in toks:
            visible |= self.hgraph.unfold_by_label_contains(t)
        if visible:
            return visible
        if self.config.default_view is not None:
            try:
                return self.hgraph.get_view(self.config.default_view)
            except KeyError:
                return None
        return None

    def answer(self, query: str, neuromod: Neuromodulators = Neuromodulators()) -> dict[str, Any]:
        decision = self.mirror.observe_and_decide(query, neuromod)
        restrict_ids = self._context_to_view(query)
        retrieved = self.index.retrieve(query, limit=self.config.top_k, restrict_to_ids=restrict_ids)
        ranked = [(g, s * decision.weight) for g, s in retrieved]
        ranked.sort(key=lambda x: x[1], reverse=True)

        match decision.strategy:
            case "imitate":
                pass
            case "adapt":
                if restrict_ids:
                    for gid in restrict_ids:
                        if all(gid != rg.id for rg, _ in ranked):
                            g = self.hgraph.nodes[gid]
                            ranked.append((g, ranked[-1][1] * 0.85 if ranked else 0.5))
                            break
            case "reject":
                ranked = ranked[:2]
            case _:
                raise RuntimeError("Unbekannte Strategie – sollte nicht auftreten.")

        synthesis = self._synthesize(query, ranked)
        return {
            "strategy": decision.strategy,
            "weight": decision.weight,
            "rationale": decision.rationale,
            "results": [{"id": g.id, "title": g.metadata.get("title", g.id), "score": s} for g, s in ranked],
            "answer": synthesis,
        }
