"""FastAPI application entrypoint for ResearchLens AI."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app import __version__
from app.api.chat import router as chat_router
from app.api.comparison import router as comparison_router
from app.api.evaluation import router as evaluation_router
from app.api.papers import router as papers_router
from app.config import settings
from app.database.metadata_store import metadata_store
from app.database.vector_store import vector_store
from app.schemas import HealthResponse
from app.services.retrieval_service import retrieval_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("researchlens.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown orchestration."""
    logger.info("Initializing ResearchLens AI backend...")
    settings.ensure_directories()
    # Pre-build / sync BM25 index with persistent store
    try:
        retrieval_service.refresh_bm25_index()
        logger.info(
            f"Initialized with {metadata_store.count_papers()} papers "
            f"and {vector_store.count_chunks()} indexed passages."
        )
    except Exception as e:
        logger.warning(f"Initial index sync warning: {e}")

    yield
    logger.info("Shutting down ResearchLens AI backend.")


app = FastAPI(
    title="ResearchLens AI",
    description="Research Paper Intelligence Platform — RAG, Multi-Paper Comparison, Summaries & Evaluation",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit frontend and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(papers_router)
app.include_router(chat_router)
app.include_router(comparison_router)
app.include_router(evaluation_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint distinguishing core application and storage availability
    from optional Groq LLM API key readiness.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        groq_configured=settings.is_groq_configured,
        groq_model=settings.groq_model,
        embedding_model=settings.embedding_model_name,
        indexed_papers_count=metadata_store.count_papers(),
        indexed_chunks_count=vector_store.count_chunks(),
        vector_db_ready=True,
        active_retrieval_mode=settings.default_retrieval_mode,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure safe error responses without leaking internal stack traces."""
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check application logs."},
    )
