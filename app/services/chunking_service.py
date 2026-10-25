"""Intelligent page-aware chunking service with section detection."""

from typing import Dict, List, Optional
from app.config import settings
from app.schemas import Chunk, ChunkMetadata
from app.utils.text_processing import detect_section_header


class ChunkingService:
    """Chunks academic papers page-by-page while tracking sections and metadata."""

    def __init__(
        self,
        default_chunk_size: Optional[int] = None,
        default_chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = default_chunk_size or settings.default_chunk_size
        self.chunk_overlap = default_chunk_overlap or settings.default_chunk_overlap

    def chunk_paper_pages(
        self,
        document_id: str,
        paper_title: str,
        pages_data: List[Dict[str, any]],
        source_filename: str,
        authors: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None,
        publication_date: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Split page-by-page text into overlapping chunks, tracking current section and page.
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap
        if overlap >= size:
            overlap = max(0, size // 4)

        authors = authors or []
        chunks: List[Chunk] = []
        global_chunk_idx = 0
        current_section = "Abstract"  # Default initial section for academic papers

        for page_item in pages_data:
            page_num = page_item["page_number"]
            page_text = page_item["text"]

            if not page_text.strip():
                continue

            # Check if any new section heading starts on this page
            lines = page_text.split("\n")
            for line in lines:
                new_sec = detect_section_header(line)
                if new_sec:
                    current_section = new_sec

            # Slide window across page_text
            step = size - overlap
            text_len = len(page_text)

            start = 0
            while start < text_len:
                end = min(start + size, text_len)
                chunk_str = page_text[start:end].strip()

                # Avoid tiny trailing fragments
                if len(chunk_str) >= 40:
                    chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                    metadata = ChunkMetadata(
                        document_id=document_id,
                        paper_title=paper_title,
                        authors=authors,
                        arxiv_id=arxiv_id,
                        source_filename=source_filename,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        section=current_section,
                        publication_date=publication_date,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=chunk_str,
                            metadata=metadata,
                        )
                    )
                    global_chunk_idx += 1

                if end == text_len:
                    break
                start += step

        return chunks


chunking_service = ChunkingService()
