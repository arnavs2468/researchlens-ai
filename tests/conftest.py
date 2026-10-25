"""Pytest configuration and shared test fixtures."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.config import settings
from app.database.metadata_store import MetadataStore
from app.database.vector_store import VectorStore
from app.main import app
from app.schemas import Chunk, ChunkMetadata, PaperMetadata, RetrievedChunk


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    """Generate a clean synthetic 2-page PDF in memory using PyMuPDF."""
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page()
    text1 = (
        "Attention Is All You Need\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on complex recurrent or "
        "convolutional neural networks. We propose the Transformer, a model architecture "
        "eschewing recurrence and relying entirely on an attention mechanism.\n"
        "Introduction\n"
        "Recurrent neural networks have been firmly established as state of the art approaches "
        "in sequence modeling and transduction problems such as language modeling and translation."
    )
    page1.insert_textbox(page1.rect, text1, fontsize=10)

    # Page 2
    page2 = doc.new_page()
    text2 = (
        "Model Architecture\n"
        "The Transformer follows this overall architecture using stacked self-attention and "
        "point-wise, fully connected layers for both the encoder and decoder.\n"
        "Experiments and Results\n"
        "On the WMT 2014 English-to-German translation task, the big transformer model outperforms "
        "the best previously reported models by more than 2.0 BLEU, establishing a new state of the art BLEU score of 28.4.\n"
        "Conclusion\n"
        "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention."
    )
    page2.insert_textbox(page2.rect, text2, fontsize=10)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def scanned_pdf_bytes():
    """Generate an empty/image-like PDF with no extractable text."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_chunks():
    """Provide realistic sample chunks for testing citation and retrieval logic."""
    return [
        RetrievedChunk(
            chunk_id="paper1_p1_c0",
            document_id="paper1",
            paper_title="Attention Is All You Need",
            page_number=1,
            chunk_index=0,
            section="Abstract",
            source_filename="transformer.pdf",
            text="The Transformer relies entirely on an attention mechanism to draw global dependencies.",
            score=0.92,
            retrieval_method="dense",
        ),
        RetrievedChunk(
            chunk_id="paper1_p2_c1",
            document_id="paper1",
            paper_title="Attention Is All You Need",
            page_number=2,
            chunk_index=1,
            section="Results",
            source_filename="transformer.pdf",
            text="On the WMT 2014 English-to-German translation task, it achieved 28.4 BLEU.",
            score=0.88,
            retrieval_method="dense",
        ),
    ]
