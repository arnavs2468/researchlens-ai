"""Research Chat view: Grounded conversational RAG with citation inspection."""

import streamlit as st
from frontend.api_client import api_client


def render_chat():
    st.markdown("## 💬 Research Assistant Chat")
    st.caption("Ask questions grounded strictly in your indexed literature. All claims feature verifiable citations with source page numbers.")

    papers = api_client.list_papers()
    if not papers:
        st.warning("⚠️ No papers are indexed yet. Please upload papers or import from arXiv before chatting.")
        return

    # Retrieval filters and configuration
    with st.expander("🛠️ Retrieval & Context Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            paper_options = {p["document_id"]: p["paper_title"] for p in papers}
            selected_docs = st.multiselect(
                "Filter by Papers (Default: All)",
                options=list(paper_options.keys()),
                format_func=lambda did: paper_options.get(did, did)[:55],
                help="Select specific papers to restrict search scope",
            )
        with c2:
            retrieval_mode = st.selectbox(
                "Retrieval Engine",
                options=["hybrid", "dense", "bm25"],
                format_func=lambda m: {"hybrid": "Hybrid (BM25 + Dense RRF)", "dense": "Dense Semantic (ChromaDB)", "bm25": "Keyword (BM25)"}[m],
                index=0,
            )
        with c3:
            top_k = st.slider("Top Passages (K)", min_value=1, max_value=15, value=5)

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Chat history display
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If message contains citations, render source cards
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 Verified Sources & Excerpts ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(
                            f"**[{cit.get('citation_id')}] {cit.get('paper_title')}** (Page {cit.get('page_number')})"
                        )
                        st.info(f"\"{cit.get('excerpt')}\"")
                        if cit.get("paper_url"):
                            st.markdown(f"🔗 [View on arXiv]({cit.get('paper_url')})")
                        st.markdown("---")

            # Warning if any citations were unvalidated
            unvalidated = msg.get("unvalidated_citations", [])
            if unvalidated:
                st.warning(f"⚠️ Unvalidated citation tags found in response: {', '.join(unvalidated)}")

    # Clear chat button
    col_empty, col_clear = st.columns([6, 1])
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    # User input
    user_query = st.chat_input("Ask a research question (e.g., 'What is the attention mechanism in Transformer?')")

    if user_query:
        # Append user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and synthesizing grounded answer..."):
                response = api_client.send_chat(
                    question=user_query,
                    document_ids=selected_docs if selected_docs else None,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    chat_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat_messages"][-4:]
                    ],
                )

                answer_text = response.get("answer", "")
                citations = response.get("citations", [])
                unvalidated = response.get("unvalidated_citations", [])
                warning = response.get("warning")

                st.markdown(answer_text)

                if warning:
                    st.caption(f"ℹ️ {warning}")

                if citations:
                    with st.expander(f"📚 Verified Sources & Excerpts ({len(citations)})", expanded=True):
                        for cit in citations:
                            st.markdown(
                                f"**[{cit.get('citation_id')}] {cit.get('paper_title')}** — Page {cit.get('page_number')}"
                            )
                            st.info(f"\"{cit.get('excerpt')}\"")
                            if cit.get("paper_url"):
                                st.markdown(f"🔗 [View Paper]({cit.get('paper_url')})")
                            st.markdown("---")

                if unvalidated:
                    st.warning(f"⚠️ Unvalidated citation tags found: {', '.join(unvalidated)}")

                # Save assistant message to session state
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                    "unvalidated_citations": unvalidated,
                })
