"""Compare Papers view: Multi-paper comparative analysis."""

import pandas as pd
import streamlit as st
from frontend.api_client import api_client


def render_compare():
    st.markdown("## ⚖️ Multi-Paper Comparative Analysis")
    st.caption("Select 2 to 5 indexed papers to generate a structured comparison matrix across methodologies, datasets, architectures, and findings.")

    papers = api_client.list_papers()
    if len(papers) < 2:
        st.warning("⚠️ At least 2 papers must be indexed to perform comparative analysis. Please upload or import more papers.")
        return

    paper_options = {p["document_id"]: p["paper_title"] for p in papers}

    selected_docs = st.multiselect(
        "Select 2 to 5 Papers for Comparison",
        options=list(paper_options.keys()),
        format_func=lambda did: f"📄 {paper_options.get(did, did)[:60]}",
        default=list(paper_options.keys())[:min(2, len(paper_options))],
        help="Select between 2 and 5 papers",
    )

    if len(selected_docs) < 2:
        st.info("Please select at least 2 papers above to begin comparison.")
        return
    if len(selected_docs) > 5:
        st.error("Maximum 5 papers can be compared at one time.")
        return

    if st.button("📊 Run Comparative Analysis", type="primary"):
        with st.spinner("Extracting comparative dimensions and generating structured analysis..."):
            res = api_client.compare_papers(document_ids=selected_docs)

            if not res.get("success"):
                st.error(f"Comparison failed: {res.get('error')}")
            else:
                st.session_state["comparison_results"] = res.get("data")

    # Render results if available
    data = st.session_state.get("comparison_results")
    if data:
        st.markdown("---")
        st.markdown("### 📋 Structured Comparison Matrix")

        paper_titles = data.get("paper_titles", {})
        table_rows = data.get("comparison_table", [])

        # Build clean pandas table for display
        df_records = []
        for row in table_rows:
            aspect = row.get("aspect", "")
            record = {"Aspect": aspect}
            values = row.get("values", {})
            for did in selected_docs:
                col_name = paper_titles.get(did, did)[:35]
                val = values.get(did, "Not found in indexed evidence")
                record[col_name] = val
            df_records.append(record)

        if df_records:
            df = pd.DataFrame(df_records)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🧠 Synthesized Comparative Analysis")

        synthesis = data.get("synthesis", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🤝 Key Similarities")
            st.write(synthesis.get("key_similarities", "Not available."))

            st.markdown("#### ⚡ Critical Differences")
            st.write(synthesis.get("critical_differences", "Not available."))

        with col2:
            st.markdown("#### ⚖️ Potential Trade-offs")
            st.write(synthesis.get("trade_offs", "Not available."))

            st.markdown("#### ❓ Open Research Questions")
            st.write(synthesis.get("open_questions", "Not available."))

        # Supporting citations
        citations = data.get("citations", [])
        if citations:
            with st.expander(f"📚 Supporting Evidence Passages ({len(citations)})"):
                for cit in citations:
                    st.markdown(
                        f"**[{cit.get('citation_id')}] {cit.get('paper_title')}** (Page {cit.get('page_number')})"
                    )
                    st.info(f"\"{cit.get('excerpt')}\"")
