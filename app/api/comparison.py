"""API endpoints for multi-paper structured comparative analysis."""

from fastapi import APIRouter, HTTPException, status
from app.schemas import CompareRequest, CompareResponse
from app.services.comparison_service import comparison_service

router = APIRouter(prefix="/papers", tags=["Paper Comparison"])


@router.post("/compare", response_model=CompareResponse)
def compare_papers(request: CompareRequest):
    """
    Compare 2 to 5 indexed papers across methodology, architectures, datasets,
    metrics, findings, limitations, and future directions.
    """
    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 papers are required for comparison.",
        )
    if len(request.document_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Comparison supports a maximum of 5 papers simultaneously.",
        )

    try:
        return comparison_service.compare_papers(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
