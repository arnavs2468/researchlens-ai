"""Pydantic schemas for request/response models across ResearchLens AI."""

from datetime import datetime
from typing import Any, Dict, List, Optional
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field


# ==============================================================================
# Document & Paper Schemas
# ==============================================================================

class PaperMetadata(BaseModel):
    document_id: str
    paper_title: str
    authors: List[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    source_filename: str
    publication_date: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChunkMetadata(BaseModel):
    document_id: str
    paper_title: str
    authors: List[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    source_filename: str
    page_number: int
    chunk_index: int
    section: Optional[str] = None
    publication_date: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    paper_title: str
    page_number: int
    chunk_index: int
    section: Optional[str] = None
    arxiv_id: Optional[str] = None
    source_filename: str
    text: str
    score: float = 0.0
    retrieval_method: str = "dense"  # "dense", "bm25", "hybrid"


# ==============================================================================
# Citations
# ==============================================================================

class CitationReference(BaseModel):
    citation_id: str  # e.g., "S1", "S2"
    document_id: str
    paper_title: str
    page_number: int
    excerpt: str
    arxiv_id: Optional[str] = None
    source_filename: str
    paper_url: Optional[str] = None
    is_validated: bool = True
    section: Optional[str] = None


# ==============================================================================
# ArXiv Integration Schemas
# ==============================================================================

class ArxivPaperResult(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: Optional[str] = None
    pdf_url: str
    entry_url: str
    categories: List[str] = Field(default_factory=list)
    is_already_indexed: bool = False


class ArxivSearchRequest(BaseModel):
    query: str
    max_results: int = 10


class ArxivImportRequest(BaseModel):
    arxiv_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    pdf_url: Optional[str] = None


# ==============================================================================
# Search & Retrieval Schemas
# ==============================================================================

class SearchQueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    retrieval_mode: str = "hybrid"  # "dense", "bm25", "hybrid"
    top_k: int = 5
    enable_reranking: bool = False


class SearchQueryResponse(BaseModel):
    query: str
    retrieval_mode: str
    total_results: int
    chunks: List[RetrievedChunk]


# ==============================================================================
# RAG Chat Schemas
# ==============================================================================

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    citations: Optional[List[CitationReference]] = None


class ChatRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    retrieval_mode: str = "hybrid"  # "dense", "bm25", "hybrid"
    top_k: int = 5
    enable_reranking: bool = False
    chat_history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationReference] = Field(default_factory=list)
    unvalidated_citations: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    is_grounded: bool = True
    model_used: str
    warning: Optional[str] = None


# ==============================================================================
# Comparison Schemas
# ==============================================================================

class CompareRequest(BaseModel):
    document_ids: List[str]
    focus_aspects: Optional[List[str]] = None


class ComparisonRow(BaseModel):
    aspect: str
    values: Dict[str, str]  # paper_id -> summary string
    citations: Dict[str, List[CitationReference]] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    document_ids: List[str]
    paper_titles: Dict[str, str]
    comparison_table: List[ComparisonRow]
    synthesis: Dict[str, str]  # similarities, differences, trade_offs, open_questions
    citations: List[CitationReference] = Field(default_factory=list)
    model_used: str
    warning: Optional[str] = None


# ==============================================================================
# Summary Schemas
# ==============================================================================

class SummaryRequest(BaseModel):
    document_id: str


class SummarySection(BaseModel):
    title: str
    content: str
    citations: List[CitationReference] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    document_id: str
    paper_title: str
    authors: List[str]
    arxiv_id: Optional[str] = None
    sections: List[SummarySection]
    markdown_content: str
    model_used: str
    warning: Optional[str] = None


# ==============================================================================
# Evaluation Schemas
# ==============================================================================

class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    category: str  # "Factual lookup", "Methodology", "Dataset and metrics", "Cross-paper comparison", "Insufficient evidence"
    target_document_ids: List[str]
    expected_answer_summary: str
    relevant_keywords: List[str] = Field(default_factory=list)
    relevant_chunk_ids: Optional[List[str]] = None
    is_unanswerable: bool = False


class MetricResult(BaseModel):
    retrieval_mode: str
    k: int
    hit_rate: float
    recall: float
    mrr: float
    precision: float
    total_evaluated: int


class QuestionEvaluationDetail(BaseModel):
    question_id: str
    question: str
    category: str
    target_document_ids: List[str]
    retrieved_chunk_doc_ids: List[str]
    hit: bool
    reciprocal_rank: float
    precision_at_k: float


class EvaluationRunRequest(BaseModel):
    retrieval_modes: List[str] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    k_values: List[int] = Field(default_factory=lambda: [3, 5, 10])
    selected_categories: Optional[List[str]] = None


class EvaluationRunResponse(BaseModel):
    timestamp: str
    benchmark_size: int
    metrics: List[MetricResult]
    details: List[QuestionEvaluationDetail]
    notes: str


# ==============================================================================
# Health & Status Schemas
# ==============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    groq_configured: bool
    groq_model: str
    embedding_model: str
    indexed_papers_count: int
    indexed_chunks_count: int
    vector_db_ready: bool
    active_retrieval_mode: str
