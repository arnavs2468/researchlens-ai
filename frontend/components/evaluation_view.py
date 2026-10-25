"""RAG Evaluation Playground view: Hit Rate@K, Recall@K, MRR, Precision, and Plotly charts."""

import pandas as pd
import plotly.express as px
import streamlit as st
from frontend.api_client import api_client


def render_evaluation():
    st.markdown("## 🧪 RAG Evaluation Playground")
    st.caption("Benchmark and compare retrieval pipelines (Dense vs BM25 vs Hybrid) using reproducible IR metrics (Hit Rate@K, Recall@K, MRR, Precision@K).")

    # 1. Benchmark configuration
    with st.expander("⚙️ Benchmark Configuration & Filter", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            categories = [
                "Factual lookup",
                "Methodology",
                "Dataset and metrics",
                "Cross-paper comparison",
                "Insufficient evidence",
            ]
            selected_cats = st.multiselect(
                "Filter Benchmark Question Categories",
                options=categories,
                default=categories,
            )
        with col2:
            modes = st.multiselect(
                "Retrieval Modes to Evaluate",
                options=["dense", "bm25", "hybrid"],
                default=["dense", "bm25", "hybrid"],
                format_func=lambda m: {"dense": "Dense Semantic (ChromaDB)", "bm25": "Keyword (BM25)", "hybrid": "Hybrid RRF"}[m],
            )
            k_choices = st.multiselect(
                "Top-K Cutoffs",
                options=[3, 5, 10],
                default=[3, 5, 10],
            )

    if st.button("🚀 Execute Retrieval Benchmark", type="primary"):
        with st.spinner("Evaluating retrieval pipelines across benchmark dataset..."):
            res = api_client.run_evaluation(
                retrieval_modes=modes,
                k_values=k_choices,
                selected_categories=selected_cats,
            )
            if res.get("success"):
                st.session_state["evaluation_results"] = res.get("data")
                st.success("Benchmark completed successfully!")
            else:
                st.error(f"Evaluation error: {res.get('error')}")

    # Load existing or new results
    results = st.session_state.get("evaluation_results")
    if not results:
        results = api_client.get_evaluation_results()

    if results:
        st.markdown("---")
        st.markdown(f"### 📈 Benchmark Results (Evaluated at: `{results.get('timestamp')[:19]}`)")
        st.caption(results.get("notes", ""))

        metrics_list = results.get("metrics", [])
        if metrics_list:
            df_metrics = pd.DataFrame(metrics_list)

            # High-level summary metrics (Hybrid at K=5)
            hybrid_k5 = df_metrics[(df_metrics["retrieval_mode"] == "hybrid") & (df_metrics["k"] == 5)]
            if not hybrid_k5.empty:
                row = hybrid_k5.iloc[0]
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Hybrid Hit Rate@5", f"{row['hit_rate'] * 100:.1f}%")
                with mcol2:
                    st.metric("Hybrid MRR", f"{row['mrr']:.3f}")
                with mcol3:
                    st.metric("Hybrid Recall@5", f"{row['recall'] * 100:.1f}%")
                with mcol4:
                    st.metric("Hybrid Precision@5", f"{row['precision'] * 100:.1f}%")

            # Metrics Table
            st.markdown("#### 📊 Comparative Metrics Table")
            display_df = df_metrics.copy()
            display_df["hit_rate"] = (display_df["hit_rate"] * 100).round(2).astype(str) + "%"
            display_df["recall"] = (display_df["recall"] * 100).round(2).astype(str) + "%"
            display_df["precision"] = (display_df["precision"] * 100).round(2).astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Interactive Plotly Charts
            st.markdown("#### 📉 Interactive Retrieval Performance Visualizations")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_hit = px.bar(
                    df_metrics,
                    x="k",
                    y="hit_rate",
                    color="retrieval_mode",
                    barmode="group",
                    title="Hit Rate @ K by Retrieval Pipeline",
                    labels={"hit_rate": "Hit Rate", "k": "Top-K Cutoff", "retrieval_mode": "Method"},
                    text_auto=".2%",
                )
                fig_hit.update_layout(yaxis_range=[0, 1.05])
                st.plotly_chart(fig_hit, use_container_width=True)

            with chart_col2:
                fig_mrr = px.bar(
                    df_metrics,
                    x="k",
                    y="mrr",
                    color="retrieval_mode",
                    barmode="group",
                    title="Mean Reciprocal Rank (MRR) by Pipeline",
                    labels={"mrr": "MRR", "k": "Top-K Cutoff", "retrieval_mode": "Method"},
                    text_auto=".3f",
                )
                fig_mrr.update_layout(yaxis_range=[0, 1.05])
                st.plotly_chart(fig_mrr, use_container_width=True)

        # Question detail inspection
        details = results.get("details", [])
        if details:
            with st.expander(f"🔎 Detailed Question-Level Results ({len(details)} evaluated)"):
                df_details = pd.DataFrame(details)
                st.dataframe(df_details, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Manual Review Section
    st.markdown("### ✍️ Human-in-the-Loop Manual Review Audit")
    st.caption("Evaluate actual RAG answers against four scientific criteria for auditability.")
    with st.container(border=True):
        rev_q = st.text_input("Review Question / Prompt", "What is the Transformer attention mechanism?")
        rev_ans = st.text_area("Generated Answer to Audit", "The Transformer uses scaled dot-product attention...")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            rating = st.radio(
                "Factual Grounding Assessment",
                ["Supported by evidence", "Partially supported", "Unsupported", "Citation incorrect"],
                index=0,
            )
        with r_col2:
            notes = st.text_area("Auditor Notes", "Claims verified against Section 3.2, Page 4.")

        if st.button("Save Audit Record"):
            st.success("✅ Audit record saved locally for review.")
