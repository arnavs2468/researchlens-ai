"""API endpoints for grounded RAG research chat."""

from fastapi import APIRouter, HTTPException, status
from app.schemas import ChatRequest, ChatResponse
from app.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["Research Chat"])


@router.post("", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    """
    Submit a research question across one or multiple indexed papers.
    Retrieves evidence passages and synthesizes an answer with validated inline citations.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty.",
        )
    return rag_service.answer_question(request)
