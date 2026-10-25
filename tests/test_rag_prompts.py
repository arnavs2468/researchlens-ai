"""Unit tests for RAG prompt construction and missing API key behavior."""

from app.config import settings
from app.prompts.answer_prompt import SYSTEM_PROMPT_RESEARCH_QA, USER_PROMPT_TEMPLATE
from app.schemas import ChatRequest
from app.services.rag_service import rag_service
from app.utils.citations import format_evidence_block


def test_prompt_template_injection(sample_chunks):
    """Verify that evidence block with [S1] tags is cleanly formatted into prompt."""
    evidence_block, citation_map = format_evidence_block(sample_chunks)
    prompt = USER_PROMPT_TEMPLATE.format(
        evidence_block=evidence_block,
        question="What is the BLEU score?",
    )

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "Attention Is All You Need" in prompt
    assert "What is the BLEU score?" in prompt
    assert "STRICT GROUNDING" in SYSTEM_PROMPT_RESEARCH_QA


def test_missing_api_key_behavior(sample_chunks, monkeypatch):
    """Test that missing GROQ_API_KEY does not crash and provides instructions."""
    # Ensure groq_api_key is None
    monkeypatch.setattr(settings, "groq_api_key", None)

    # Mock retrieval service to return sample chunks
    monkeypatch.setattr(
        "app.services.rag_service.retrieval_service.retrieve",
        lambda **kwargs: sample_chunks,
    )

    request = ChatRequest(question="What is self-attention?")
    response = rag_service.answer_question(request)

    assert response.retrieval_mode == "hybrid"
    assert len(response.retrieved_chunks) == 2
    assert "GROQ_API_KEY Not Configured" in response.answer
    assert "console.groq.com" in response.answer
    assert response.warning is not None
