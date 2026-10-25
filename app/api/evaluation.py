"""API endpoints for reproducible RAG retrieval benchmarking."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.schemas import (
    BenchmarkQuestion,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from app.services.evaluation_service import evaluation_service

router = APIRouter(prefix="/evaluation", tags=["RAG Evaluation"])


@router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation(request: EvaluationRunRequest):
    """
    Run retrieval evaluation across Dense, BM25, and Hybrid RRF modes.
    Computes Hit Rate@K, Recall@K, MRR, and Precision@K.
    """
    return evaluation_service.run_evaluation(request)


@router.get("/results", response_model=Optional[EvaluationRunResponse])
def get_latest_results():
    """Retrieve the most recent benchmark evaluation results."""
    results = evaluation_service.get_latest_results()
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation runs found. Execute /evaluation/run first.",
        )
    return results


@router.get("/questions", response_model=List[BenchmarkQuestion])
def get_benchmark_questions(category: Optional[str] = Query(None)):
    """List benchmark questions, optionally filtered by category."""
    return evaluation_service.get_benchmark_questions(category=category)


@router.post("/questions", response_model=BenchmarkQuestion)
def add_benchmark_question(question: BenchmarkQuestion):
    """Add or update a question in the evaluation benchmark dataset."""
    return evaluation_service.add_benchmark_question(question)
