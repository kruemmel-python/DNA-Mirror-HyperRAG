"""Streamlit application entry point for DNA-Mirror-HyperRAG."""

from __future__ import annotations

from typing import Any, Iterable

import streamlit as st

from dna_mirror_hyperrag.core import Neuromodulators
from dna_mirror_hyperrag.runtime import (
    DEFAULT_PROMOTERS,
    RuntimeState,
    ingest_text_document,
    initialize_runtime,
)


@st.cache_resource(show_spinner=False)
def load_runtime() -> RuntimeState:
    return initialize_runtime()


def _decode_uploaded_file(uploaded_file: Any) -> str:
    data = uploaded_file.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - UI guard
        raise ValueError(
            f"Datei {uploaded_file.name!r} ist keine UTF-8 Textdatei."
        ) from exc


def _render_results(result: dict[str, object]) -> None:
    st.subheader("Antwort")
    st.markdown(
        f"**Strategie:** {result['strategy']}  |  **Gewichtung:** {result['weight']:.2f}"
    )
    st.caption(result["rationale"])
    st.markdown(f"**Quantum Jump Factor:** {result['quantum_jump_factor']:.3f}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Synthese")
        st.write(result["answer"])
    with cols[1]:
        st.markdown("### Ergebnisse")
        for entry in result["results"]:
            st.markdown(
                f"- `{entry['id']}` · **{entry.get('title', entry['id'])}** — Score: {entry['score']:.3f}"
            )


def main() -> None:
    st.set_page_config(page_title="DNA-Mirror-HyperRAG", layout="wide")
    runtime = load_runtime()
    hg, rag = runtime.graph, runtime.rag

    st.sidebar.header("Neuromodulatoren")
    dopamin = st.sidebar.slider("Dopamin", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    serotonin = st.sidebar.slider("Serotonin", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    gaba = st.sidebar.slider("GABA", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    st.sidebar.header("Upload")
    tokens_per_chunk = st.sidebar.number_input(
        "Tokens pro Chunk", min_value=50, max_value=600, value=220, step=10
    )
    overlap = st.sidebar.number_input("Overlap", min_value=0, max_value=200, value=40, step=5)
    promoters = st.sidebar.multiselect(
        "Standard-Promoter",
        options=[f"{name}:{boost}" for name, boost in DEFAULT_PROMOTERS],
        default=[f"{name}:{boost}" for name, boost in DEFAULT_PROMOTERS],
        help="Wird angewendet, wenn Texte in Gene umgewandelt werden.",
    )

    st.title("DNA-Mirror-HyperRAG Streamlit")
    st.write(
        "Interagiere mit dem DNA-Mirror-HyperRAG, stelle Fragen und erweitere das"
        " Wissensnetz direkt aus Textdateien."
    )

    st.markdown(
        f"Aktueller Wissensgraph: **{len(hg.nodes)} Gene** · **{len(hg.edges)} Kanten**"
    )

    with st.expander("📚 Textdateien hochladen", expanded=False):
        uploaded_files = st.file_uploader(
            "Wähle eine oder mehrere Textdateien", type=["txt", "md"], accept_multiple_files=True
        )
        if uploaded_files and st.button("Dateien integrieren", use_container_width=True):
            added_total = []
            added_edges = []
            selected_promoters: Iterable[tuple[str, float]] = []
            if promoters:
                selected_promoters = []
                for entry in promoters:
                    name, raw_boost = entry.split(":", 1)
                    selected_promoters.append((name, float(raw_boost)))
            else:
                selected_promoters = DEFAULT_PROMOTERS

            for uploaded in uploaded_files:
                try:
                    text = _decode_uploaded_file(uploaded)
                except ValueError as exc:
                    st.error(str(exc))
                    continue

                result = ingest_text_document(
                    text,
                    uploaded.name or "upload",
                    runtime,
                    tokens_per_chunk=int(tokens_per_chunk),
                    overlap=int(overlap),
                    promoters=selected_promoters,
                )
                added_total.extend(result["added_genes"])
                if result["edge_id"]:
                    added_edges.append(result["edge_id"])

            if added_total:
                st.success(
                    f"{len(added_total)} Gene und {len(added_edges)} Hyperkanten hinzugefügt."
                )
            else:
                st.warning("Keine verwertbaren Textinhalte gefunden.")

    st.subheader("🔍 Anfrage stellen")
    query = st.text_area(
        "Deine Frage", value="Erkläre DNA-basiertes RAG mit k-Mer Hybridisierung", height=140
    )
    if st.button("Antwort generieren", use_container_width=True):
        if not query.strip():
            st.warning("Bitte gib eine Frage ein.")
        else:
            neuromodulators = Neuromodulators(dopamin, serotonin, gaba)
            with st.spinner("Berechne Antwort..."):
                result = rag.answer(query, neuromod=neuromodulators)
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        _render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()

