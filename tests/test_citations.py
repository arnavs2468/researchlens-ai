"""Unit tests for citation extraction and validation."""

from app.utils.citations import (
    extract_citations_from_text,
    format_evidence_block,
    validate_and_build_citations,
)


def test_extract_citations_patterns():
    """Verify regex patterns for single, multiple, and bracket-separated citations."""
    text1 = "The Transformer uses self-attention [S1] and achieves 28.4 BLEU [S2]."
    assert extract_citations_from_text(text1) == ["S1", "S2"]

    text2 = "Both models outperform baselines [S1, S2, S3]."
    assert extract_citations_from_text(text2) == ["S1", "S2", "S3"]

    text3 = "Consecutive citations [S1][S2]."
    assert extract_citations_from_text(text3) == ["S1", "S2"]


def test_validate_and_build_citations(sample_chunks):
    """Verify that cited IDs matching the evidence block are validated and mapped."""
    evidence_block, citation_map = format_evidence_block(sample_chunks)

    assert "S1" in citation_map
    assert "S2" in citation_map

    # Model generates answer citing valid S1, S2 and an invented S99
    generated_text = (
        "The model uses self-attention mechanisms [S1] and scored 28.4 BLEU [S2]. "
        "Also it claims 99% accuracy [S99]."
    )

    validated, unvalidated = validate_and_build_citations(generated_text, citation_map)

    assert len(validated) == 2
    assert validated[0].citation_id == "S1"
    assert validated[0].page_number == 1
    assert validated[1].citation_id == "S2"
    assert validated[1].page_number == 2

    # S99 was not in citation_map and must be flagged as unvalidated
    assert unvalidated == ["S99"]
