"""Unit tests for PDF extraction and validation."""

import pytest
from app.services.pdf_service import PDFProcessingError, pdf_service


def test_pdf_extraction_success(sample_pdf_bytes):
    """Test successful extraction of title, pages, and metadata from valid PDF."""
    title, pages_data, meta = pdf_service.extract_paper(sample_pdf_bytes, "sample.pdf")

    assert "Attention Is All You Need" in title
    assert len(pages_data) == 2
    assert pages_data[0]["page_number"] == 1
    assert "Transformer" in pages_data[0]["text"]
    assert pages_data[1]["page_number"] == 2
    assert "BLEU" in pages_data[1]["text"]
    assert meta["total_pages"] == 2
    assert len(meta["file_hash"]) == 64  # SHA-256


def test_scanned_pdf_rejection(scanned_pdf_bytes):
    """Test that scanned / blank PDFs are rejected with informative error message."""
    with pytest.raises(PDFProcessingError) as exc_info:
        pdf_service.extract_paper(scanned_pdf_bytes, "scanned.pdf")

    assert "scanned document or image-only PDF" in str(exc_info.value)
    assert "Optical Character Recognition (OCR) is not currently supported" in str(exc_info.value)


def test_invalid_pdf_header():
    """Test rejection of non-PDF bytes."""
    with pytest.raises(PDFProcessingError) as exc_info:
        pdf_service.extract_paper(b"Not a real pdf content", "fake.pdf")

    assert "missing %PDF- header" in str(exc_info.value)
