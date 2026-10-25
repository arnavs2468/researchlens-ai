"""ResearchLens AI — Main Streamlit Application."""

import streamlit as st
from frontend.api_client import api_client
from frontend.components.chat_view import render_chat
from frontend.components.compare_view import render_compare
from frontend.components.dashboard_view import render_dashboard
from frontend.components.discover_view import render_discover
from frontend.components.evaluation_view import render_evaluation
from frontend.components.settings_view import render_settings
from frontend.components.summary_view import render_summary
from frontend.components.upload_view import render_upload

# Configure Streamlit page
st.set_page_config(
    page_title="ResearchLens AI — Paper Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic aesthetic, typography, and clean spacing
st.markdown(
    """
    <style>
    /* Main body background and font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Clean subtle metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Elegant containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 16px;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Options
NAVIGATION_OPTIONS = [
    "Dashboard",
    "Discover Papers",
    "Upload Papers",
    "Research Chat",
    "Compare Papers",
    "Paper Summaries",
    "RAG Evaluation",
    "Settings",
]

# Sidebar Brand & Navigation
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔬 ResearchLens AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Research Paper Intelligence Platform</div>", unsafe_allow_html=True)

    # Manage navigation via session state
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Dashboard"

    current_idx = (
        NAVIGATION_OPTIONS.index(st.session_state["nav_selection"])
        if st.session_state["nav_selection"] in NAVIGATION_OPTIONS
        else 0
    )

    selected_nav = st.radio(
        "Navigation",
        options=NAVIGATION_OPTIONS,
        index=current_idx,
        label_visibility="collapsed",
    )

    if selected_nav != st.session_state["nav_selection"]:
        st.session_state["nav_selection"] = selected_nav

    st.markdown("---")

    # Quick system badge in sidebar
    health = api_client.get_health()
    if health.get("status") == "healthy":
        st.caption("🟢 Backend: **Online**")
        if health.get("groq_configured"):
            st.caption(f"⚡ Groq LLM: **Ready** (`{health.get('groq_model')}`)")
        else:
            st.caption("ℹ️ Groq LLM: **Key Not Set** (Retrieval only)")
        st.caption(f"📚 Indexed Papers: **{health.get('indexed_papers_count', 0)}**")
    else:
        st.caption("🔴 Backend: **Offline**")

    st.caption("---")
    st.caption("Developed with FastAPI • ChromaDB • Groq • Streamlit")

# Route to selected view
view = st.session_state["nav_selection"]

if view == "Dashboard":
    render_dashboard()
elif view == "Discover Papers":
    render_discover()
elif view == "Upload Papers":
    render_upload()
elif view == "Research Chat":
    render_chat()
elif view == "Compare Papers":
    render_compare()
elif view == "Paper Summaries":
    render_summary()
elif view == "RAG Evaluation":
    render_evaluation()
elif view == "Settings":
    render_settings()
