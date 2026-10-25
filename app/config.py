"""Configuration settings for ResearchLens AI."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Groq API Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"

    # Server Configuration
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_port: int = 8501

    # Storage Paths
    data_dir: Path = BASE_DIR / "data"
    papers_storage_dir: Path = BASE_DIR / "data" / "papers"
    vector_db_dir: Path = BASE_DIR / "data" / "chroma_db"
    metadata_db_path: Path = BASE_DIR / "data" / "papers_metadata.json"
    evaluation_dir: Path = BASE_DIR / "data" / "evaluation"

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Chunking
    default_chunk_size: int = 800
    default_chunk_overlap: int = 150

    # Retrieval
    default_retrieval_mode: str = "hybrid"  # "dense", "bm25", "hybrid"
    default_top_k: int = 5
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ArXiv defaults
    arxiv_max_results: int = 10

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers_storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_groq_configured(self) -> bool:
        """Return True if a non-placeholder GROQ_API_KEY is present."""
        return bool(
            self.groq_api_key
            and self.groq_api_key.strip()
            and not self.groq_api_key.strip().startswith("your_")
        )


settings = Settings()
settings.ensure_directories()
