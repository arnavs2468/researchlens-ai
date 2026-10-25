"""Paper Summaries view: 8-section grounded academic summarization."""

import streamlit as st
from frontend.api_client import api_client


def render_summary():
    st.markdown("## 📑 Structured Paper Summarization")
    st.caption("Generate grounded, comprehensive 8-section academic summaries. Export the complete report as Markdown.")

    papers = api_client.list_papers()
    if not papers:
        st.warning("⚠️ No papers indexed yet. Please upload papers or import from arXiv first.")
        return

    paper_options = {p["document_id"]: p["paper_title"] for p in papers}

    # Check if a document was pre-selected from dashboard
    default_doc = st.session_state.get("selected_summary_doc_id")
    if default_doc not in paper_options:
        default_doc = list(paper_options.keys())[0]

    selected_doc_id = st.selectbox(
        "Choose Paper to Summarize",
        options=list(paper_options.keys()),
        index=list(paper_options.keys()).index(default_doc),
        format_func=lambda did: f"📄 {paper_options.get(did, did)}",
    )

    if st.button("✨ Generate Structured Summary", type="primary"):
        with st.spinner("Extracting sections and generating structured summary..."):
            res = api_client.get_summary(selected_doc_id)
            if not res.get("success"):
                st.error(f"Failed to generate summary: {res.get('error')}")
            else:
                st.session_state["active_summary"] = res.get("data")

    # Render active summary
    summary_data = st.session_state.get("active_summary")
    if summary_data and summary_data.get("document_id") == selected_doc_id:
        st.markdown("---")
        head_col, download_col = st.columns([5, 1.5])
        with head_col:
            st.markdown(f"### {summary_data.get('paper_title')}")
            authors = ", ".join(summary_data.get("authors", [])) if summary_data.get("authors") else "Not listed"
            st.caption(f"✍️ **Authors:** {authors} | 🏷️ **Model:** `{summary_data.get('model_used')}`")
        with download_col:
            md_content = summary_data.get("markdown_content", "")
            clean_filename = f"summary_{selected_doc_id}.md"
            st.download_button(
                label="📥 Export Markdown",
                data=md_content,
                file_name=clean_filename,
                mime="text/markdown",
                use_container_width=True,
            )

        # Render the 8 sections
        sections = summary_data.get("sections", [])
        for sec in sections:
            with st.container(border=True):
                st.markdown(f"#### 📌 {sec.get('title')}")
                st.write(sec.get("content"))

                citations = sec.get("citations", [])
                if citations:
                    with st.expander(f"🔍 Section Sources ({len(citations)})"):
                        for c in citations:
                            st.caption(f"**[{c.get('citation_id')}]** Page {c.get('page_number')}: \"{c.get('excerpt')}\"")
