
"""
dna_mirror_hyperrag.loaders
Markdown- und JSON-Loader + Graph-Builder für HyperGraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import json, re
from pathlib import Path

from dna_mirror_hyperrag.core import Gene, RegulatorySite, HyperEdge, HyperGraph


@dataclass
class MarkdownSectionRule:
    heading_pattern: str = r'^##\s+(.*)$'
    as_topic_edge: bool = True

def load_markdown_as_genes(md_path: str | Path, rule: MarkdownSectionRule = MarkdownSectionRule(),
                           default_promoters: list[tuple[str, float]] | None = None):
    p = Path(md_path)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    heading_re = re.compile(rule.heading_pattern, re.IGNORECASE)

    genes: list[Gene] = []
    edges: list[HyperEdge] = []
    current_title: str | None = None
    buf: list[str] = []
    idx = 0

    def flush():
        nonlocal idx, buf, current_title
        if not buf: return
        seq = "\n".join(buf).strip()
        if not seq:
            buf = []; return
        gid = f"MD_{p.stem}_{idx}"
        sites = [RegulatorySite("promoter", t, b) for (t,b) in (default_promoters or [])]
        genes.append(Gene(gid, seq, sites=sites, metadata={"title": current_title or gid, "source": str(p)}))
        if rule.as_topic_edge and current_title:
            edges.append(HyperEdge(f"E_{gid}", current_title, {gid}))
        idx += 1
        buf = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            current_title = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return genes, edges

def load_json_kb(json_path: str | Path, text_field: str = "text", title_field: str = "title",
                 regulatory: dict[str, list[tuple[str, float]]] | None = None):
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON muss eine Liste von Objekten enthalten.")
    genes: list[Gene] = []
    edges: list[HyperEdge] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict): continue
        seq = str(item.get(text_field, "")).strip()
        if not seq: continue
        title = str(item.get(title_field, f"JSON_{p.stem}_{i}"))
        tags = item.get("tags", [])
        sites = []
        if regulatory and isinstance(tags, list):
            for tag in tags:
                for pat, boost in regulatory.get(tag, []):
                    sites.append(RegulatorySite("promoter", pat, float(boost)))
        gid = f"JSON_{p.stem}_{i}"
        genes.append(Gene(gid, seq, sites=sites, metadata={"title": title, "source": str(p), "tags": tags}))
        if tags:
            edges.append(HyperEdge(f"E_{gid}", " ".join(map(str, tags)), {gid}))
    return genes, edges

def build_hgraph_from_sources(md_files: Iterable[str | Path] = (), json_files: Iterable[str | Path] = (),
                              default_promoters: list[tuple[str, float]] | None = None,
                              json_regulatory: dict[str, list[tuple[str, float]]] | None = None) -> HyperGraph:
    hg = HyperGraph()
    # Markdown
    for md in md_files:
        genes, edges = load_markdown_as_genes(md, default_promoters=default_promoters)
        for g in genes: hg.add_gene(g)
        for e in edges:
            if e.id in hg.edges:
                hg.edges[e.id].members |= e.members
            else:
                hg.add_edge(e)
    # JSON
    for jf in json_files:
        genes, edges = load_json_kb(jf, regulatory=json_regulatory)
        for g in genes: hg.add_gene(g)
        for e in edges:
            if e.id in hg.edges:
                hg.edges[e.id].members |= e.members
            else:
                hg.add_edge(e)
    # Standard-View
    first = list(hg.nodes.keys())[:6]
    if first:
        hg.make_view("grundlagen", set(first))
    return hg
