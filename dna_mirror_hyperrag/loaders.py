
"""
dna_mirror_hyperrag.loaders
---------------------------
Einfache Loader für Markdown- und JSON-Wissensquellen.
Sie erzeugen Gene + optionale HyperEdges (Topics) aus Dateien.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any
import json
import re
from pathlib import Path

from dna_mirror_hyperrag.core import Gene, RegulatorySite, HyperEdge, HyperGraph


# -----------------------------
# Markdown-Loader
# -----------------------------

@dataclass
class MarkdownSectionRule:
    """
    Regel zur Extraktion von Abschnitten.
    - heading_pattern: Regex zur Identifikation von Überschriften (z. B. r'^##\\s+(.*)$')
    - as_topic_edge:   Erzeuge eine HyperEdge pro Heading-Block (optional)
    """
    heading_pattern: str = r'^##\s+(.*)$'
    as_topic_edge: bool = True


def load_markdown_as_genes(
    md_path: str | Path,
    rule: MarkdownSectionRule = MarkdownSectionRule(),
    default_promoters: list[tuple[str, float]] | None = None
) -> tuple[list[Gene], list[HyperEdge]]:
    """
    Lädt ein Markdown-Dokument und zerlegt es in Abschnitte.
    Jeder Abschnitt wird ein Gene. Optional entstehen HyperEdges pro Überschrift.
    """
    p = Path(md_path)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()

    heading_re = re.compile(rule.heading_pattern, flags=re.IGNORECASE)
    genes: list[Gene] = []
    edges: list[HyperEdge] = []

    current_title: str | None = None
    current_buffer: list[str] = []
    section_idx = 0

    def flush_section() -> None:
        nonlocal section_idx, current_title, current_buffer
        if not current_buffer:
            return
        content = "\n".join(current_buffer).strip()
        if not content:
            current_buffer = []
            return
        gid = f"MD_{p.stem}_{section_idx}"
        sites = [RegulatorySite("promoter", t, b) for (t, b) in (default_promoters or [])]
        genes.append(Gene(id=gid, sequence=content, sites=sites, metadata={"title": current_title or gid, "source": str(p)}))
        if rule.as_topic_edge and current_title:
            edges.append(HyperEdge(id=f"E_{gid}", label=current_title, members={gid}))
        section_idx += 1
        current_buffer = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush_section()
            current_title = m.group(1).strip()
        else:
            current_buffer.append(line)

    flush_section()
    return genes, edges


# -----------------------------
# JSON-Loader
# -----------------------------

def load_json_kb(
    json_path: str | Path,
    text_field: str = "text",
    title_field: str = "title",
    regulatory: dict[str, list[tuple[str, float]]] | None = None,
) -> tuple[list[Gene], list[HyperEdge]]:
    """
    Erwartet JSON mit einer Liste von Objekten, z. B.:
      [
        {"title": "Intro", "text": "....", "tags": ["neuro"]},
        {"title": "DNA",   "text": "...."}
      ]

    'regulatory' kann pro Tag eine Liste von (pattern, boost) definieren.
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON-Wissensbasis muss eine Liste von Objekten enthalten.")

    genes: list[Gene] = []
    edges: list[HyperEdge] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        seq = str(item.get(text_field, "")).strip()
        if not seq:
            continue
        title = str(item.get(title_field, f"JSON_{p.stem}_{i}"))
        tags = item.get("tags", [])
        sites = []
        if regulatory and isinstance(tags, list):
            for tag in tags:
                for pat, boost in regulatory.get(tag, []):
                    sites.append(RegulatorySite("promoter", pat, float(boost)))
        gid = f"JSON_{p.stem}_{i}"
        genes.append(Gene(id=gid, sequence=seq, sites=sites, metadata={"title": title, "source": str(p), "tags": tags}))
        if tags:
            edges.append(HyperEdge(id=f"E_{gid}", label=" ".join(map(str, tags)), members={gid}))

    return genes, edges


# -----------------------------
# Graph Builder
# -----------------------------

def build_hgraph_from_sources(
    md_files: Iterable[str | Path] = (),
    json_files: Iterable[str | Path] = (),
    default_promoters: list[tuple[str, float]] | None = None,
    json_regulatory: dict[str, list[tuple[str, float]]] | None = None,
) -> HyperGraph:
    """
    Baut einen HyperGraph aus Markdown- und JSON-Quellen.
    """
    hg = HyperGraph()
    # Markdown
    for md in md_files:
        genes, edges = load_markdown_as_genes(md, default_promoters=default_promoters)
        for g in genes:
            hg.add_gene(g)
        for e in edges:
            # Verbinde gleichnamige Topics, falls wiederkehrend
            if e.id in hg.edges:
                # zusammenführen
                existing = hg.edges[e.id]
                existing.members |= e.members
            else:
                hg.add_edge(e)

    # JSON
    for jf in json_files:
        genes, edges = load_json_kb(jf, regulatory=json_regulatory)
        for g in genes:
            hg.add_gene(g)
        for e in edges:
            if e.id in hg.edges:
                existing = hg.edges[e.id]
                existing.members |= e.members
            else:
                hg.add_edge(e)

    # Beispiel: Standard-View „grundlagen“ (erste 6 Gene)
    first_ids = list(hg.nodes.keys())[:6]
    if first_ids:
        hg.make_view("grundlagen", set(first_ids))
    return hg
