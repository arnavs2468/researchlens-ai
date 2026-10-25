"""Unit tests for intelligent chunking and section detection."""

from app.services.chunking_service import chunking_service
from app.utils.text_processing import detect_section_header


def test_section_header_detection():
    """Verify regex patterns for standard academic section headings."""
    assert detect_section_header("Abstract") == "Abstract"
    assert detect_section_header("1. Introduction") == "Introduction"
    assert detect_section_header("3. Methodology") == "Methodology"
    assert detect_section_header("Experimental Setup") == "Experiments"
    assert detect_section_header("Results and Findings") == "Results"
    assert detect_section_header("Limitations") == "Limitations"
    assert detect_section_header("5. Conclusion") == "Conclusion"
    # Sentence ending in period should NOT be classified as header
    assert detect_section_header("This is an introduction to deep learning.") is None


def test_chunking_preserves_pages_and_sections():
    """Verify that chunks retain page numbers, section tags, and no empty chunks are made."""
    pages_data = [
        {
            "page_number": 1,
            "text": "Abstract\nWe introduce a new architecture.\nIntroduction\nDeep neural networks require scale.",
        },
        {
            "page_number": 2,
            "text": "Experiments\nWe evaluated our approach on standard translation benchmarks and scored high BLEU.",
        },
    ]

    chunks = chunking_service.chunk_paper_pages(
        document_id="test_doc",
        paper_title="Test Paper",
        pages_data=pages_data,
        source_filename="test.pdf",
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) >= 2
    # Check page numbers
    page_numbers = {c.metadata.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers

    # Ensure no empty or tiny chunks
    for c in chunks:
        assert len(c.text.strip()) >= 40
        assert c.metadata.document_id == "test_doc"
        assert c.metadata.paper_title == "Test Paper"
        assert c.metadata.section is not None
