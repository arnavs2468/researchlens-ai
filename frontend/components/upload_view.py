"""Upload view component for PDF file ingestion and corpus management."""

import streamlit as st
from frontend.api_client import api_client


def render_upload():
    st.markdown("## 📤 Ingest Research Papers")
    st.caption("Upload scientific PDF documents for page-by-page text extraction, intelligent chunking, and local vector indexing.")

    # Ingestion Configuration
    with st.expander("⚙️ Advanced Chunking Configuration", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.slider(
                "Chunk Size (Characters)",
                min_value=300,
                max_value=2000,
                value=800,
                step=50,
                help="Target character length for each passage chunk",
            )
        with c2:
            chunk_overlap = st.slider(
                "Chunk Overlap (Characters)",
                min_value=50,
                max_value=400,
                value=150,
                step=25,
                help="Overlap between adjacent chunks to maintain context continuity",
            )

    uploaded_files = st.file_uploader(
        "Select Academic PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload non-scanned PDFs up to 60MB each.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s) for ingestion.")
        if st.button("🚀 Ingest & Index Selected Papers", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            files_payload = []
            for f in uploaded_files:
                files_payload.append(("files", (f.name, f.getvalue(), "application/pdf")))

            status_text.text("Extracting text, computing embeddings, and updating ChromaDB...")
            progress_bar.progress(40)

            result = api_client.upload_pdfs(
                files=files_payload,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            progress_bar.progress(100)

            if result.get("success"):
                papers = result.get("papers", [])
                st.success(f"🎉 Successfully indexed {len(papers)} paper(s)!")
                for p in papers:
                    st.write(
                        f"✅ **{p.get('paper_title')}** — {p.get('total_pages')} pages, {p.get('total_chunks')} chunks"
                    )
            else:
                st.error(f"Ingestion failed: {result.get('error')}")

    st.markdown("---")

    # Corpus management: List and delete indexed papers
    st.markdown("### 🗃️ Currently Indexed Corpus")
    papers = api_client.list_papers()

    if not papers:
        st.info("No papers are currently indexed.")
    else:
        st.write(f"Total Papers in Database: **{len(papers)}**")
        for p in papers:
            with st.container(border=True):
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    st.markdown(f"**{p.get('paper_title')}**")
                    authors = ", ".join(p.get("authors", [])) if p.get("authors") else "Unspecified"
                    st.caption(
                        f"ID: `{p.get('document_id')}` | Authors: {authors} | "
                        f"Pages: {p.get('total_pages')} | Chunks: {p.get('total_chunks')}"
                    )
                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{p.get('document_id')}", type="secondary"):
                        with st.spinner("Deleting paper and embeddings..."):
                            del_res = api_client.delete_paper(p.get("document_id"))
                            if del_res.get("success"):
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error(del_res.get("error"))
