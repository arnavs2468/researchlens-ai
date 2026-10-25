"""Dashboard view component for ResearchLens AI."""

import streamlit as st
from frontend.api_client import api_client


def render_dashboard():
    st.markdown("## 📊 Research Overview & Corpus Status")
    st.caption("Central dashboard for monitoring indexed literature, retrieval mode, and system health.")

    health = api_client.get_health()
    is_online = health.get("status") == "healthy"

    if not is_online:
        st.error(
            "⚠️ Backend service is currently unreachable at `http://127.0.0.1:8000`. "
            "Please start the FastAPI backend using `python scripts/run_backend.py`."
        )
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Indexed Papers",
            value=health.get("indexed_papers_count", 0),
            help="Total research papers indexed in local storage",
        )
    with col2:
        st.metric(
            label="Vector Passages",
            value=health.get("indexed_chunks_count", 0),
            help="Total passages embedded in local ChromaDB",
        )
    with col3:
        st.metric(
            label="Active Retrieval",
            value=health.get("active_retrieval_mode", "hybrid").upper(),
            help="Default retrieval pipeline (Dense + BM25 via RRF)",
        )
    with col4:
        groq_ready = health.get("groq_configured", False)
        status_label = "Ready (Free Tier)" if groq_ready else "Key Not Set"
        st.metric(
            label="LLM Synthesis",
            value=status_label,
            delta="Online" if groq_ready else "Retrieval Only",
            delta_color="normal" if groq_ready else "off",
            help="Groq LLM API configuration status",
        )

    st.markdown("---")

    # Quick action banner
    st.markdown("### ⚡ Quick Actions")
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("🔍 Discover Papers on arXiv", use_container_width=True):
            st.session_state["nav_selection"] = "Discover Papers"
            st.rerun()
    with qcol2:
        if st.button("📤 Upload Research PDFs", use_container_width=True):
            st.session_state["nav_selection"] = "Upload Papers"
            st.rerun()
    with qcol3:
        if st.button("💬 Ask Research Chat", use_container_width=True):
            st.session_state["nav_selection"] = "Research Chat"
            st.rerun()

    st.markdown("---")

    # Recent Papers section
    st.markdown("### 📚 Indexed Papers")
    papers = api_client.list_papers()

    if not papers:
        st.info(
            "No research papers indexed yet. "
            "Click **Discover Papers** to import papers from arXiv, or **Upload Papers** to index your own PDFs."
        )
    else:
        for p in papers[:6]:
            with st.container(border=True):
                head_col, action_col = st.columns([5, 1])
                with head_col:
                    st.markdown(f"#### {p.get('paper_title', 'Untitled')}")
                    authors_str = ", ".join(p.get("authors", [])) if p.get("authors") else "Authors not specified"
                    st.caption(f"✍️ {authors_str} | 📄 Pages: {p.get('total_pages', 0)} | 🧩 Chunks: {p.get('total_chunks', 0)}")
                    if p.get("arxiv_id"):
                        st.markdown(f"🔗 [arXiv:{p.get('arxiv_id')}](https://arxiv.org/abs/{p.get('arxiv_id')})")
                with action_col:
                    if st.button("Summarize", key=f"sum_{p.get('document_id')}"):
                        st.session_state["selected_summary_doc_id"] = p.get("document_id")
                        st.session_state["nav_selection"] = "Paper Summaries"
                        st.rerun()

    # System Architecture Overview
    st.markdown("---")
    st.markdown("### 🏛️ Retrieval & RAG Pipeline Architecture")
    st.markdown(
        """
        - **Local Embedding Engine:** Sentence-Transformers `all-MiniLM-L6-v2` (384-dimensional embeddings generated locally with zero API cost).
        - **Vector Store:** ChromaDB persistent local database storing normalized document vectors.
        - **Lexical Index:** BM25Okapi keyword index synchronizing across all paper passages.
        - **Hybrid Retrieval:** Reciprocal Rank Fusion (RRF with $k=60$) combining semantic dense retrieval with BM25 keyword matching to prevent vocabulary mismatch.
        - **LLM Synthesis:** Groq API using free-tier high-speed models (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
        - **Strict Citation Grounding:** Answers are validated against retrieved passages, mapping stable identifiers `[S1]`, `[S2]` to exact pages and excerpts.
        """
    )
