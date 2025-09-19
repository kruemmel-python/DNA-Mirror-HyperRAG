
"""
dna_mirror_hyperrag.core
Kern: DNA-Index (k-Mer Hybridisierung), Hypergraph (Folding/Unfolding),
Mirror-Neurologie (Imitate/Adapt/Reject mit Neuromodulatoren),
RAG-Pipeline (Observe -> Retrieve -> Gate -> Synthesize). Python 3.12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, Callable
from collections import defaultdict, Counter
import math
import random


# ---------------- Hilfsfunktionen ----------------

def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())

def _tokenize(s: str) -> list[str]:
    return _normalize_whitespace(s).lower().split()

def _kmerize(tokens: Sequence[str], k: int) -> list[tuple[str, ...]]:
    if k <= 0:
        raise ValueError(f"k muss > 0 sein, erhalten: {k}")
    return [tuple(tokens[i:i+k]) for i in range(0, max(0, len(tokens)-k+1))]

def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t]*b[t] for t in set(a)&set(b))
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0


# ---------------- DNA-Regulatorik & Gene ----------------

@dataclass(frozen=True)
class RegulatorySite:
    type: str          # "promoter" | "enhancer" | "silencer"
    pattern: str
    boost: float = 0.0

@dataclass
class Gene:
    id: str
    sequence: str
    sites: list[RegulatorySite] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    energy: float = 0.5

    def tokens(self) -> list[str]:
        return _tokenize(self.sequence)

    def bow(self) -> Counter[str]:
        return Counter(self.tokens())


# ---------------- Hypergraph ----------------

@dataclass
class HyperEdge:
    id: str
    label: str
    members: set[str]
    folded: bool = False

@dataclass
class HyperGraph:
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
        e = self.edges.get(edge_id)
        if not e: raise KeyError(f"Unbekannte HyperEdge: {edge_id!r}")
        e.folded = True

    def unfold(self, edge_id: str) -> None:
        e = self.edges.get(edge_id)
        if not e: raise KeyError(f"Unbekannte HyperEdge: {edge_id!r}")
        e.folded = False

    def unfold_by_label_contains(self, pattern: str) -> set[str]:
        patt = pattern.lower()
        visible: set[str] = set()
        for e in self.edges.values():
            if patt in e.label.lower():
                e.folded = False
                visible |= e.members
        return visible

    def make_view(self, name: str, ids: set[str]) -> None:
        missing = [gid for gid in ids if gid not in self.nodes]
        if missing: raise ValueError(f"View {name!r} enthält unbekannte Gene: {missing}")
        self.views[name] = set(ids)

    def get_view(self, name: str) -> set[str]:
        if name not in self.views:
            raise KeyError(f"Unbekannte View: {name!r}")
        return set(self.views[name])


# ---------------- DNA-Index ----------------

class LightEnergyModule:
    def __init__(
        self,
        decay_rate: float = 0.005,
        recharge_amount: float = 0.1,
        min_energy: float = 0.1,
    ):
        self.gene_energies: dict[str, float] = defaultdict(lambda: 0.5)
        self.decay_rate = decay_rate
        self.recharge_amount = recharge_amount
        self.min_energy = min_energy

    def set_initial_energy(self, gene_id: str, value: float = 0.5):
        if gene_id not in self.gene_energies:
            self.gene_energies[gene_id] = max(self.min_energy, min(1.0, value))

    def decay_all_energies(self, gene_ids: Iterable[str]):
        for gid in gene_ids:
            current = self.gene_energies.get(gid, 0.5)
            self.gene_energies[gid] = max(self.min_energy, current - self.decay_rate)

    def recharge_energy(self, gene_id: str, amount: float | None = None):
        current_amount = amount if amount is not None else self.recharge_amount
        current = self.gene_energies.get(gene_id, self.min_energy)
        self.gene_energies[gene_id] = max(
            self.min_energy, min(1.0, current + current_amount)
        )

    def get_energy(self, gene_id: str) -> float:
        return self.gene_energies.get(gene_id, self.min_energy)


class QuantumFluctuationModule:
    def __init__(
        self,
        jump_threshold: float = 0.7,
        plasma_influence: float = 0.5,
        max_jump_factor: float = 0.3,
    ):
        self.jump_threshold = jump_threshold
        self.plasma_influence = plasma_influence
        self.max_jump_factor = max_jump_factor

    def generate_jump_factor(self, energy_field: float, plasma_field: float) -> float:
        energy_field = max(0.0, min(1.0, energy_field))
        plasma_field = max(0.0, min(1.0, plasma_field))
        overlap = (energy_field + plasma_field * self.plasma_influence) / (
            1.0 + self.plasma_influence
        )
        if overlap > self.jump_threshold:
            jump_magnitude = (overlap - self.jump_threshold) / (
                1.0 - self.jump_threshold
            )
            return random.uniform(
                -self.max_jump_factor * jump_magnitude,
                self.max_jump_factor * jump_magnitude,
            )
        return random.uniform(
            -self.max_jump_factor * 0.1, self.max_jump_factor * 0.1
        )


@dataclass
class DNAIndex:
    k: int = 3
    postings: dict[tuple[str, ...], set[str]] = field(default_factory=lambda: defaultdict(set))
    genes: dict[str, Gene] = field(default_factory=dict)
    energy_module: LightEnergyModule | None = None

    def add(self, gene: Gene) -> None:
        if gene.id in self.genes:
            raise ValueError(f"Gene-ID bereits vorhanden: {gene.id!r}")
        self.genes[gene.id] = gene
        for km in _kmerize(gene.tokens(), self.k):
            self.postings[km].add(gene.id)
        if self.energy_module:
            self.energy_module.set_initial_energy(gene.id, gene.energy)

    def build_from(
        self, genes: Iterable[Gene], energy_module: LightEnergyModule | None = None
    ) -> "DNAIndex":
        if energy_module is not None:
            self.energy_module = energy_module
        for g in genes:
            self.add(g)
        return self

    def _reg_boost(self, gene: Gene, query: str) -> float:
        q = _normalize_whitespace(query).lower()
        boost = 0.0
        for s in gene.sites:
            pat = s.pattern.lower()
            if pat and pat in q:
                match s.type.lower():
                    case "promoter": boost += abs(s.boost) + 0.2
                    case "enhancer": boost += abs(s.boost)
                    case "silencer": boost -= abs(s.boost)
                    case _: boost += 0.0
        return boost

    def _hyb(self, q_tokens: list[str], gene: Gene) -> float:
        qk = _kmerize(q_tokens, self.k)
        if not qk: return 0.0
        gk = set(_kmerize(gene.tokens(), self.k))
        matches = sum(1 for km in qk if km in gk)
        return matches/len(qk)

    def _bow_bonus(self, q_tokens: list[str], gene: Gene) -> float:
        return 0.15 * _cosine(Counter(q_tokens), gene.bow())

    def retrieve(self, query: str, limit: int = 5, restrict_to_ids: set[str] | None = None) -> list[tuple[Gene, float]]:
        if not query or not query.strip():
            raise ValueError("Leere Anfrage kann nicht abgerufen werden.")
        q_tokens = _tokenize(query)

        cand: set[str] = set()
        for km in _kmerize(q_tokens, self.k):
            cand |= self.postings.get(km, set())
        if not cand:
            cand = set(self.genes.keys())

        if restrict_to_ids is not None:
            cand &= restrict_to_ids
            if not cand:
                cand = restrict_to_ids

        scored: list[tuple[Gene, float]] = []
        for gid in cand:
            g = self.genes[gid]
            energy_boost = 0.0
            if self.energy_module:
                gene_energy = self.energy_module.get_energy(gid)
                energy_boost = (gene_energy - 0.5) * 0.5
            score = max(
                0.0,
                self._hyb(q_tokens, g)
                + self._reg_boost(g, query)
                + self._bow_bonus(q_tokens, g)
                + energy_boost,
            )
            scored.append((g, score))
            if self.energy_module:
                self.energy_module.recharge_energy(gid)

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# ---------------- Mirror-Neurologie ----------------

@dataclass
class Neuromodulators:
    dopamin: float = 1.0
    serotonin: float = 1.0
    gaba: float = 1.0
    def clamp(self) -> "Neuromodulators":
        def c(x: float) -> float: return max(0.0, min(2.0, x))
        return Neuromodulators(c(self.dopamin), c(self.serotonin), c(self.gaba))

@dataclass
class MirrorDecision:
    strategy: str
    weight: float
    rationale: str
    clarity: float
    novelty: float
    risk: float

class MirrorModule:
    def __init__(self, policy: Callable[[str], dict[str, float]] | None = None):
        self.policy = policy or self._default_policy

    def _default_policy(self, query: str) -> dict[str, float]:
        q = _normalize_whitespace(query).lower()
        toks = _tokenize(q)
        novelty = min(1.0, len(set(toks))/12)
        risk = 0.8 if any(w in q for w in ("hack","exploit","bypass","waffe","angreifen","malware")) else 0.1
        clarity = 0.7 if len(q) >= 20 else 0.4
        return {"novelty": novelty, "risk": risk, "clarity": clarity}

    def observe_and_decide(self, query: str, neuromod: Neuromodulators = Neuromodulators()) -> MirrorDecision:
        if not query or not query.strip():
            raise ValueError("MirrorModule: Leere Anfrage kann nicht beobachtet werden.")
        feats = self.policy(query)
        m = neuromod.clamp()
        imitate = 0.6*feats["clarity"] + 0.3*(1-feats["novelty"]) + 0.1*m.dopamin
        adapt   = 0.5*feats["novelty"] + 0.3*feats["clarity"] + 0.2*m.serotonin
        reject  = 0.6*feats["risk"] + 0.2*(1-feats["clarity"]) + 0.2*m.gaba
        strategy = max(
            (("imitate", imitate), ("adapt", adapt), ("reject", reject)),
            key=lambda x: x[1],
        )[0]
        match strategy:
            case "imitate":
                return MirrorDecision(
                    "imitate",
                    1.0 + 0.5 * m.dopamin,
                    "Klares bekanntes Muster → Imitation.",
                    feats["clarity"],
                    feats["novelty"],
                    feats["risk"],
                )
            case "adapt":
                return MirrorDecision(
                    "adapt",
                    1.0 + 0.2 * m.serotonin,
                    "Teilweise neuartig → vorsichtige Adaption.",
                    feats["clarity"],
                    feats["novelty"],
                    feats["risk"],
                )
            case "reject":
                return MirrorDecision(
                    "reject",
                    0.5,
                    "Risiko/Unklarheit → dämpfen und beschneiden.",
                    feats["clarity"],
                    feats["novelty"],
                    feats["risk"],
                )
            case _:
                raise RuntimeError("Unbekannte Mirror-Strategie.")
        

# ---------------- RAG-Pipeline ----------------

@dataclass
class RAGConfig:
    top_k: int = 5
    synthesis_max_sentences: int = 4
    kmer_k: int = 3
    default_view: str | None = None
    dynamic_k: bool = True  # Auto-k: kurz→2, mittel→3, lang→4
    energy_decay_rate: float = 0.005
    energy_recharge_amount: float = 0.1
    min_gene_energy: float = 0.1
    quantum_jump_threshold: float = 0.7
    quantum_plasma_influence: float = 0.5
    quantum_max_jump_factor: float = 0.3

class DNAMirrorHyperRAG:
    def __init__(self, hgraph: HyperGraph, config: RAGConfig = RAGConfig()):
        self.hgraph = hgraph
        self.config = config
        self.light_energy_module = LightEnergyModule(
            decay_rate=config.energy_decay_rate,
            recharge_amount=config.energy_recharge_amount,
            min_energy=config.min_gene_energy,
        )
        self.index = DNAIndex(k=config.kmer_k).build_from(
            hgraph.nodes.values(), energy_module=self.light_energy_module
        )
        self.mirror = MirrorModule()
        self.quantum_fluctuation_module = QuantumFluctuationModule(
            jump_threshold=config.quantum_jump_threshold,
            plasma_influence=config.quantum_plasma_influence,
            max_jump_factor=config.quantum_max_jump_factor,
        )

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
        self.light_energy_module.decay_all_energies(self.hgraph.nodes.keys())
        restrict_ids = self._context_to_view(query)

        # Dynamisches k: kurzer Text -> k=2, mittel -> 3, lang -> 4
        if self.config.dynamic_k:
            qlen = len(query.split())
            new_k = 2 if qlen <= 4 else (3 if qlen <= 12 else 4)
            if new_k != self.index.k:
                tmp_index = DNAIndex(k=new_k).build_from(
                    self.hgraph.nodes.values(), energy_module=self.light_energy_module
                )
                retrieved = tmp_index.retrieve(query, limit=self.config.top_k, restrict_to_ids=restrict_ids)
            else:
                retrieved = self.index.retrieve(query, limit=self.config.top_k, restrict_to_ids=restrict_ids)
        else:
            retrieved = self.index.retrieve(query, limit=self.config.top_k, restrict_to_ids=restrict_ids)

        ranked = [(g, s * decision.weight) for g, s in retrieved]
        energy_field_level = max(0.0, min(1.0, 1.0 - (decision.weight - 0.5)))
        plasma_field_level = decision.novelty
        quantum_jump_factor = self.quantum_fluctuation_module.generate_jump_factor(
            energy_field=energy_field_level,
            plasma_field=plasma_field_level,
        )
        ranked = [(g, max(0.0, s + quantum_jump_factor)) for g, s in ranked]
        ranked.sort(key=lambda x: x[1], reverse=True)

        match decision.strategy:
            case "imitate":
                pass
            case "adapt":
                if restrict_ids:
                    for gid in restrict_ids:
                        if all(gid != rg.id for rg, _ in ranked):
                            g = self.hgraph.nodes[gid]
                            ranked.append((g, ranked[-1][1]*0.85 if ranked else 0.5))
                            break
            case "reject":
                ranked = ranked[:2]
            case _:
                raise RuntimeError("Unbekannte Strategie.")

        synthesis = self._synthesize(query, ranked)
        return {
            "strategy": decision.strategy,
            "weight": decision.weight,
            "rationale": decision.rationale,
            "quantum_jump_factor": quantum_jump_factor,
            "results": [{"id": g.id, "title": g.metadata.get("title", g.id), "score": s} for g, s in ranked],
            "answer": synthesis,
        }
