"""Script to start the ResearchLens AI FastAPI backend."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting ResearchLens AI Backend on http://{settings.backend_host}:{settings.backend_port}...")
    print(f"Interactive API Documentation: http://{settings.backend_host}:{settings.backend_port}/docs")
    uvicorn.run(
        "app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )
