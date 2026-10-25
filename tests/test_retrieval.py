"""Unit tests for dense, BM25, and hybrid RRF retrieval logic."""

from app.schemas import RetrievedChunk
from app.services.retrieval_service import retrieval_service


def test_bm25_tokenization():
    """Verify BM25 word tokenization."""
    tokens = retrieval_service._tokenize("Attention is all you need: 2017 Transformer!")
    assert "attention" in tokens
    assert "transformer" in tokens
    assert "2017" in tokens


def test_chunk_deduplication():
    """Verify that identical or overlapping chunk texts are deduplicated."""
    chunk1 = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        paper_title="Paper",
        page_number=1,
        chunk_index=0,
        source_filename="f.pdf",
        text="The Transformer relies entirely on an attention mechanism to model dependencies.",
        score=0.9,
    )
    # Duplicate with same beginning
    chunk2 = RetrievedChunk(
        chunk_id="c2",
        document_id="d1",
        paper_title="Paper",
        page_number=1,
        chunk_index=1,
        source_filename="f.pdf",
        text="The Transformer relies entirely on an attention mechanism to model dependencies across all words.",
        score=0.85,
    )
    # Distinct chunk
    chunk3 = RetrievedChunk(
        chunk_id="c3",
        document_id="d1",
        paper_title="Paper",
        page_number=2,
        chunk_index=2,
        source_filename="f.pdf",
        text="Experimental results show that the model achieves 28.4 BLEU on translation.",
        score=0.8,
    )

    deduped = retrieval_service._deduplicate_chunks([chunk1, chunk2, chunk3])
    assert len(deduped) == 2
    assert deduped[0].chunk_id == "c1"
    assert deduped[1].chunk_id == "c3"
