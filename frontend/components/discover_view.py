"""Discover Papers view: Search and import papers from arXiv API."""

import streamlit as st
from frontend.api_client import api_client


def render_discover():
    st.markdown("## 🔍 Discover & Import Research Papers")
    st.caption("Search arXiv for peer-reviewed papers and preprints, inspect abstracts, and import them directly into your local vector index.")

    # Search bar
    with st.form("arxiv_search_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Search Topic, Title, Author, or Keyword",
                placeholder="e.g., 'Attention Is All You Need', 'RAG evaluation', 'LoRA fine-tuning'",
            )
        with col2:
            max_results = st.number_input("Max Results", min_value=1, max_value=25, value=6)

        submit = st.form_submit_button("Search arXiv", use_container_width=True)

    if submit and query:
        with st.spinner(f"Querying arXiv API for '{query}'..."):
            results = api_client.search_arxiv(query=query, max_results=max_results)
            st.session_state["arxiv_search_results"] = results

    results = st.session_state.get("arxiv_search_results", [])

    if results:
        st.markdown(f"### Found {len(results)} Papers on arXiv")

        for idx, paper in enumerate(results):
            with st.container(border=True):
                col_content, col_action = st.columns([5, 1.2])

                with col_content:
                    st.markdown(f"#### {paper.get('title')}")
                    authors_str = ", ".join(paper.get("authors", [])) if paper.get("authors") else "Unknown"
                    st.caption(f"✍️ **Authors:** {authors_str} | 📅 **Published:** {paper.get('published', '')[:10]}")

                    with st.expander("📖 Read Abstract"):
                        st.write(paper.get("abstract", "No abstract available."))

                    st.markdown(
                        f"🔗 [arXiv Abstract Page]({paper.get('entry_url')}) | 📄 [Direct PDF Link]({paper.get('pdf_url')})"
                    )

                with col_action:
                    is_indexed = paper.get("is_already_indexed", False)
                    if is_indexed:
                        st.success("✅ Indexed")
                    else:
                        button_key = f"import_btn_{paper.get('arxiv_id')}_{idx}"
                        if st.button("📥 Import Paper", key=button_key, use_container_width=True):
                            with st.spinner(f"Downloading and indexing arXiv:{paper.get('arxiv_id')}..."):
                                res = api_client.import_arxiv_paper(
                                    arxiv_id=paper.get("arxiv_id"),
                                    title=paper.get("title"),
                                    authors=paper.get("authors"),
                                    pdf_url=paper.get("pdf_url"),
                                )
                                if res.get("success"):
                                    st.success(f"Indexed successfully! ({res['paper'].get('total_chunks')} chunks)")
                                    paper["is_already_indexed"] = True
                                    st.rerun()
                                else:
                                    st.error(f"Import failed: {res.get('error')}")
    elif submit and not query:
        st.warning("Please enter a search query.")
    elif "arxiv_search_results" in st.session_state and not results:
        st.info("No matching papers found on arXiv. Try refining your keywords.")
