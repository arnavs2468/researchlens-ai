"""API endpoints for managing research papers (upload, arXiv import, list, delete)."""

import io
import re
import uuid
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from app.config import settings
from app.database.metadata_store import metadata_store
from app.database.vector_store import vector_store
from app.schemas import (
    ArxivImportRequest,
    ArxivPaperResult,
    PaperMetadata,
    SearchQueryRequest,
    SearchQueryResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.services.arxiv_service import arxiv_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.pdf_service import PDFProcessingError, pdf_service
from app.services.retrieval_service import retrieval_service
from app.services.summary_service import summary_service
from app.utils.text_processing import compute_file_hash

router = APIRouter(prefix="/papers", tags=["Papers"])


@router.get("", response_model=List[PaperMetadata])
def list_papers():
    """List all indexed research papers sorted by creation time."""
    return metadata_store.list_papers()


@router.get("/{document_id}", response_model=PaperMetadata)
def get_paper(document_id: str):
    """Retrieve metadata for a specific indexed paper."""
    paper = metadata_store.get_paper(document_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID '{document_id}' not found.",
        )
    return paper


@router.post("/upload", response_model=List[PaperMetadata])
async def upload_papers(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    """
    Upload one or more research paper PDFs.
    Extracts text, splits into page-aware chunks, creates local embeddings,
    and indexes into persistent ChromaDB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_papers: List[PaperMetadata] = []
    errors: List[str] = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        try:
            content = await file.read()
            if not content:
                errors.append(f"'{filename}': File is empty.")
                continue

            file_hash = compute_file_hash(content)

            # Check duplicate by hash
            existing = metadata_store.find_by_hash(file_hash)
            if existing:
                processed_papers.append(existing)
                continue

            # Extract PDF
            extracted_title, pages_data, meta_dict = pdf_service.extract_paper(content, filename)

            # Stable document ID
            clean_title_slug = re.sub(r"[^a-zA-Z0-9]+", "-", extracted_title.lower()).strip("-")[:40]
            short_id = uuid.uuid4().hex[:6]
            document_id = f"{clean_title_slug}-{short_id}" if clean_title_slug else f"paper-{short_id}"

            # Save local copy of PDF
            saved_pdf_path = settings.papers_storage_dir / f"{document_id}.pdf"
            with open(saved_pdf_path, "wb") as f:
                f.write(content)

            # Chunk paper
            chunks = chunking_service.chunk_paper_pages(
                document_id=document_id,
                paper_title=extracted_title,
                pages_data=pages_data,
                source_filename=filename,
                authors=meta_dict.get("authors", []),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                errors.append(f"'{filename}': Could not generate valid chunks.")
                continue

            # Embed and index into ChromaDB
            vector_store.add_chunks(chunks)

            # Register metadata
            paper_meta = PaperMetadata(
                document_id=document_id,
                paper_title=extracted_title,
                authors=meta_dict.get("authors", []),
                source_filename=filename,
                total_pages=meta_dict.get("total_pages", len(pages_data)),
                total_chunks=len(chunks),
                file_path=str(saved_pdf_path),
                file_hash=file_hash,
            )
            metadata_store.add_paper(paper_meta)
            processed_papers.append(paper_meta)

        except PDFProcessingError as e:
            errors.append(f"'{filename}': {str(e)}")
        except Exception as e:
            errors.append(f"'{filename}': Unexpected error: {str(e)}")

    # Update BM25 index after ingestion
    if processed_papers:
        retrieval_service.refresh_bm25_index()

    if not processed_papers and errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors),
        )

    return processed_papers


@router.get("/arxiv/search", response_model=List[ArxivPaperResult])
async def search_arxiv(
    query: str = Query(..., min_length=2, description="Search term, topic, or author"),
    max_results: int = Query(10, ge=1, le=25),
):
    """Search arXiv by topic, title, author, or keyword."""
    return await arxiv_service.search_papers(query=query, max_results=max_results)


@router.post("/import-arxiv", response_model=PaperMetadata)
async def import_arxiv_paper(
    request: ArxivImportRequest,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """
    Import a paper directly from arXiv:
    Downloads the PDF, extracts text, chunks, embeds, and indexes into ChromaDB.
    """
    clean_arxiv_id = request.arxiv_id.strip()

    # Check if already imported
    existing = metadata_store.find_by_arxiv_id(clean_arxiv_id)
    if existing:
        return existing

    try:
        pdf_bytes = await arxiv_service.download_paper_pdf(
            clean_arxiv_id, pdf_url=request.pdf_url
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch arXiv PDF: {str(e)}",
        )

    filename = f"arxiv_{clean_arxiv_id.replace('/', '_')}.pdf"
    file_hash = compute_file_hash(pdf_bytes)

    # Check if same content was uploaded under another name
    existing_hash = metadata_store.find_by_hash(file_hash)
    if existing_hash:
        return existing_hash

    try:
        extracted_title, pages_data, meta_dict = pdf_service.extract_paper(pdf_bytes, filename)
    except PDFProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prefer requested title if provided
    final_title = request.title.strip() if request.title and request.title.strip() else extracted_title
    final_authors = request.authors if request.authors else meta_dict.get("authors", [])

    # Document ID
    clean_title_slug = re.sub(r"[^a-zA-Z0-9]+", "-", final_title.lower()).strip("-")[:35]
    document_id = f"arxiv-{clean_arxiv_id.replace('.', '-')}-{clean_title_slug}"

    # Save PDF locally
    saved_pdf_path = settings.papers_storage_dir / f"{document_id}.pdf"
    with open(saved_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Chunking
    chunks = chunking_service.chunk_paper_pages(
        document_id=document_id,
        paper_title=final_title,
        pages_data=pages_data,
        source_filename=filename,
        authors=final_authors,
        arxiv_id=clean_arxiv_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not generate valid text chunks from this arXiv paper.",
        )

    # Index into ChromaDB
    vector_store.add_chunks(chunks)

    # Save metadata
    paper_meta = PaperMetadata(
        document_id=document_id,
        paper_title=final_title,
        authors=final_authors,
        arxiv_id=clean_arxiv_id,
        source_filename=filename,
        total_pages=meta_dict.get("total_pages", len(pages_data)),
        total_chunks=len(chunks),
        file_path=str(saved_pdf_path),
        file_hash=file_hash,
        pdf_url=request.pdf_url or f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf",
    )
    metadata_store.add_paper(paper_meta)

    # Refresh BM25
    retrieval_service.refresh_bm25_index()

    return paper_meta


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_paper(document_id: str):
    """Delete a paper and all its indexed chunks."""
    paper = metadata_store.get_paper(document_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{document_id}' not found.",
        )

    # Remove from ChromaDB
    vector_store.delete_paper_chunks(document_id)

    # Remove PDF file if stored
    if paper.file_path:
        try:
            import os
            if os.path.exists(paper.file_path):
                os.remove(paper.file_path)
        except Exception:
            pass

    # Remove from metadata store
    metadata_store.delete_paper(document_id)

    # Refresh BM25
    retrieval_service.refresh_bm25_index()

    return {"message": f"Paper '{paper.paper_title}' deleted successfully.", "document_id": document_id}


@router.post("/search", response_model=SearchQueryResponse)
def search_indexed_chunks(request: SearchQueryRequest):
    """
    Search indexed passages using Dense, BM25, or Hybrid (RRF) retrieval.
    Does not require a Groq API key.
    """
    chunks = retrieval_service.retrieve(
        query=request.query,
        retrieval_mode=request.retrieval_mode,
        top_k=request.top_k,
        document_ids=request.document_ids,
        enable_reranking=request.enable_reranking,
    )
    return SearchQueryResponse(
        query=request.query,
        retrieval_mode=request.retrieval_mode,
        total_results=len(chunks),
        chunks=chunks,
    )


@router.post("/{document_id}/summary", response_model=SummaryResponse)
def get_paper_summary(document_id: str):
    """Generate structured 8-section summary of an indexed paper."""
    paper = metadata_store.get_paper(document_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper '{document_id}' not found.",
        )
    return summary_service.summarize_paper(SummaryRequest(document_id=document_id))
