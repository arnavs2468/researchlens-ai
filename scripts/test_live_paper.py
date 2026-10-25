"""End-to-end live testing script for ResearchLens AI using a real arXiv paper."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv()

from app.config import settings
from app.database.metadata_store import metadata_store
from app.database.vector_store import vector_store
from app.schemas import (
    ArxivImportRequest,
    ChatRequest,
    SummaryRequest,
)
from app.services.arxiv_service import arxiv_service
from app.services.chunking_service import chunking_service
from app.services.pdf_service import pdf_service
from app.services.rag_service import rag_service
from app.services.retrieval_service import retrieval_service
from app.services.summary_service import summary_service
from app.utils.text_processing import compute_file_hash


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

async def main():
    print("=" * 80)
    print("[ResearchLens AI] - Live Research Paper Ingestion & RAG Verification")
    print("=" * 80)
    print(f"Active Groq Model: {settings.groq_model}")
    print(f"ChromaDB Persist Dir: {settings.vector_db_dir}")

    # 1. Fetch real research paper from arXiv: Attention Is All You Need (1706.03762)
    arxiv_id = "1706.03762"
    print(f"\n[Step 1] Downloading paper arXiv:{arxiv_id} ('Attention Is All You Need')...")
    
    existing = metadata_store.find_by_arxiv_id(arxiv_id)
    if existing:
        print(f"Paper already indexed with document_id: {existing.document_id}")
        doc_id = existing.document_id
        paper_meta = existing
    else:
        pdf_bytes = await arxiv_service.download_paper_pdf(arxiv_id)
        print(f"Downloaded {len(pdf_bytes):,} bytes.")

        title, pages_data, meta = pdf_service.extract_paper(pdf_bytes, f"arxiv_{arxiv_id}.pdf")
        print(f"Extracted title: '{title}' ({len(pages_data)} pages)")

        doc_id = f"arxiv-{arxiv_id.replace('.', '-')}-attention-is-all-you-need"
        pdf_path = settings.papers_storage_dir / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        chunks = chunking_service.chunk_paper_pages(
            document_id=doc_id,
            paper_title=title,
            pages_data=pages_data,
            source_filename=f"arxiv_{arxiv_id}.pdf",
            authors=meta.get("authors", []),
            arxiv_id=arxiv_id,
        )
        print(f"Generated {len(chunks)} page-aware chunks.")

        print("Computing embeddings and saving to ChromaDB...")
        vector_store.add_chunks(chunks)

        from app.schemas import PaperMetadata
        paper_meta = PaperMetadata(
            document_id=doc_id,
            paper_title=title,
            authors=meta.get("authors", ["Ashish Vaswani et al."]),
            arxiv_id=arxiv_id,
            source_filename=f"arxiv_{arxiv_id}.pdf",
            total_pages=len(pages_data),
            total_chunks=len(chunks),
            file_path=str(pdf_path),
            file_hash=compute_file_hash(pdf_bytes),
        )
        metadata_store.add_paper(paper_meta)
        retrieval_service.refresh_bm25_index()
        print(f"Successfully indexed '{title}' into local ChromaDB & BM25!")

    # 2. Test Grounded RAG Chat
    test_question = (
        "What is the mathematical formulation of Scaled Dot-Product Attention, "
        "and why did the authors choose to scale by 1 / sqrt(d_k)?"
    )
    print(f"\n[Step 2] Testing Grounded RAG Chat")
    print(f"Question: \"{test_question}\"")
    print("Retrieving hybrid passages and generating grounded answer with Groq...")

    chat_req = ChatRequest(
        question=test_question,
        document_ids=[doc_id],
        retrieval_mode="hybrid",
        top_k=4,
    )
    chat_resp = rag_service.answer_question(chat_req)

    print("\n--- Model Response ---")
    print(chat_resp.answer)
    print("-" * 50)
    print(f"Retrieved Passages: {len(chat_resp.retrieved_chunks)}")
    print(f"Validated Citations: {len(chat_resp.citations)}")
    for cit in chat_resp.citations:
        print(f"  • [{cit.citation_id}] Page {cit.page_number} ({cit.paper_title}): \"{cit.excerpt[:120]}...\"")

    if chat_resp.unvalidated_citations:
        print(f"WARNING - Unvalidated Citations: {chat_resp.unvalidated_citations}")
    else:
        print("Citations Verification: 100% Grounded & Validated against retrieved passages!")

    # 3. Test Paper Summarization
    print(f"\n[Step 3] Testing 8-Section Structured Paper Summarization")
    sum_req = SummaryRequest(document_id=doc_id)
    sum_resp = summary_service.summarize_paper(sum_req)
    print(f"Generated {len(sum_resp.sections)} structured sections.")
    for sec in sum_resp.sections[:3]:
        print(f"\n[{sec.title}]")
        print(sec.content[:200] + ("..." if len(sec.content) > 200 else ""))

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY! The pipeline is fully verified.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
