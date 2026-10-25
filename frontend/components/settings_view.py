"""Settings view component for environment inspection and configuration guidance."""

import streamlit as st
from frontend.api_client import api_client


def render_settings():
    st.markdown("## ⚙️ System Settings & Environment Configuration")
    st.caption("Inspect active configuration, local storage directories, and Groq free-tier setup.")

    health = api_client.get_health()

    # System Status Card
    with st.container(border=True):
        st.markdown("### 🖥️ Backend Runtime Status")
        status_val = health.get("status", "unknown")
        if status_val == "healthy":
            st.success("🟢 FastAPI Backend is Online and Healthy")
        else:
            st.error(f"🔴 Backend is Offline: {health.get('error', 'Unknown')}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Application Version:** `{health.get('version', '1.0.0')}`")
            st.write(f"**Embedding Model:** `{health.get('embedding_model', 'all-MiniLM-L6-v2')}`")
        with c2:
            st.write(f"**ChromaDB Status:** `{'Ready' if health.get('vector_db_ready') else 'Not Ready'}`")
            st.write(f"**Active Retrieval Mode:** `{health.get('active_retrieval_mode', 'hybrid')}`")
        with c3:
            st.write(f"**Indexed Papers:** `{health.get('indexed_papers_count', 0)}`")
            st.write(f"**Indexed Passages:** `{health.get('indexed_chunks_count', 0)}`")

    st.markdown("---")

    # Groq API Configuration
    st.markdown("### 🔑 Groq LLM Free-Tier Configuration")
    groq_ready = health.get("groq_configured", False)

    if groq_ready:
        st.success(f"✅ Groq API Key is configured and ready. Active Model: `{health.get('groq_model')}`")
    else:
        st.warning(
            "⚠️ Groq API Key is not configured. Document ingestion, local embedding generation, "
            "and dense/BM25/hybrid passage retrieval function completely offline. "
            "To enable grounded AI answer generation, configure your free API key below."
        )

    with st.expander("📝 Step-by-Step Instructions to Set Up Free Groq API Key", expanded=not groq_ready):
        st.markdown(
            """
            1. **Sign up for a free Groq account** at [https://console.groq.com](https://console.groq.com).
            2. **Generate an API Key** under **API Keys** -> **Create API Key**.
            3. In your project directory, edit or create `.env`:
               ```bash
               GROQ_API_KEY=gsk_your_actual_key_here
               GROQ_MODEL=llama-3.3-70b-versatile
               ```
            4. Supported free-tier models:
               - `llama-3.3-70b-versatile` (Default recommended: high accuracy & reasoning)
               - `llama-3.1-8b-instant` (Fastest latency)
               - `mixtral-8x7b-32768` (High context)
            5. Restart the backend server for changes to take effect.
            """
        )

    st.markdown("---")

    # Local Storage & Privacy
    st.markdown("### 🛡️ Privacy & Local Execution Guarantee")
    st.markdown(
        """
        - **100% Local Embeddings:** All vector representations are calculated directly on your machine via `sentence-transformers/all-MiniLM-L6-v2`. No embeddings are sent to external APIs.
        - **100% Local Vector Database:** Chunks are stored locally in ChromaDB at `data/chroma_db`.
        - **Secure API Key Handling:** Secrets are never hardcoded or sent to the frontend.
        - **No Paid Subscriptions Required:** The platform runs entirely on open-source local models and Groq's free tier.
        """
    )
