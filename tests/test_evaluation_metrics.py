"""Unit tests for information retrieval evaluation metrics (Hit Rate, MRR, Recall, Precision)."""

from app.schemas import BenchmarkQuestion, RetrievedChunk
from app.services.evaluation_service import evaluation_service


def test_evaluation_metric_calculations():
    """Verify Hit Rate, Reciprocal Rank, Precision, and Recall math."""
    q = BenchmarkQuestion(
        id="q1",
        question="What is the Transformer architecture?",
        category="Methodology",
        target_document_ids=["transformer-doc"],
        expected_answer_summary="Attention based model.",
        relevant_keywords=["attention", "transformer"],
    )

    # Case 1: First retrieved chunk is relevant -> Hit=True, RR=1.0, Prec=0.5
    chunks_top_hit = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="transformer-doc",
            paper_title="Transformer",
            page_number=1,
            chunk_index=0,
            source_filename="f.pdf",
            text="The Transformer relies entirely on attention.",
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="unrelated-doc",
            paper_title="Other",
            page_number=1,
            chunk_index=0,
            source_filename="f2.pdf",
            text="Unrelated text about biology.",
        ),
    ]

    is_hit, rr, rec, prec, doc_ids = evaluation_service._evaluate_single_query(
        q, chunks_top_hit, k=2
    )

    assert is_hit is True
    assert rr == 1.0  # 1 / 1
    assert rec == 1.0
    assert prec == 0.5  # 1 relevant out of 2

    # Case 2: Second retrieved chunk is relevant -> Hit=True, RR=0.5
    chunks_second_hit = [chunks_top_hit[1], chunks_top_hit[0]]
    is_hit2, rr2, rec2, prec2, _ = evaluation_service._evaluate_single_query(
        q, chunks_second_hit, k=2
    )
    assert is_hit2 is True
    assert rr2 == 0.5  # 1 / 2

    # Case 3: No relevant chunks -> Hit=False, RR=0.0
    chunks_miss = [chunks_top_hit[1]]
    is_hit3, rr3, rec3, prec3, _ = evaluation_service._evaluate_single_query(
        q, chunks_miss, k=1
    )
    assert is_hit3 is False
    assert rr3 == 0.0
    assert prec3 == 0.0
