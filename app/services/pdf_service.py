"""PDF extraction service using PyMuPDF."""

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pymupdf  # PyMuPDF
from app.utils.text_processing import clean_extracted_text, compute_file_hash


class PDFProcessingError(Exception):
    """Custom exception for PDF parsing and extraction errors."""
    pass


class PDFService:
    """Service for processing research paper PDFs page-by-page."""

    MAX_FILE_SIZE_MB = 60

    @classmethod
    def validate_pdf_bytes(cls, content: bytes, filename: str) -> None:
        """Validate size and PDF signature."""
        size_mb = len(content) / (1024 * 1024)
        if size_mb > cls.MAX_FILE_SIZE_MB:
            raise PDFProcessingError(
                f"File '{filename}' exceeds maximum allowed size of {cls.MAX_FILE_SIZE_MB}MB."
            )
        if not content.startswith(b"%PDF-"):
            raise PDFProcessingError(
                f"File '{filename}' is not a valid PDF (missing %PDF- header)."
            )

    @classmethod
    def extract_paper(
        cls,
        content: bytes,
        filename: str,
    ) -> Tuple[str, List[Dict[str, any]], Dict[str, any]]:
        """
        Extract page text and metadata from PDF bytes.
        Returns:
            (title, pages_data, metadata_dict)
            where pages_data is a list of {'page_number': int, 'text': str}
        """
        cls.validate_pdf_bytes(content, filename)

        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as e:
            raise PDFProcessingError(f"Could not open PDF '{filename}': {str(e)}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise PDFProcessingError(f"PDF '{filename}' contains 0 pages.")

        pages_data: List[Dict[str, any]] = []
        total_extracted_chars = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text("text") or ""
            cleaned = clean_extracted_text(raw_text)
            total_extracted_chars += len(cleaned)

            pages_data.append({
                "page_number": page_idx + 1,
                "text": cleaned,
            })

        # Graceful handling for scanned / image-only PDFs
        if total_extracted_chars < 50:
            doc.close()
            raise PDFProcessingError(
                f"PDF '{filename}' appears to be a scanned document or image-only PDF "
                f"(extracted fewer than 50 characters across {total_pages} pages). "
                "Optical Character Recognition (OCR) is not currently supported."
            )

        # Extract title heuristics
        pdf_metadata = doc.metadata or {}
        extracted_title = cls._heuristic_title_extraction(pages_data, pdf_metadata, filename)

        # Extract authors heuristic from metadata if present
        authors_meta = pdf_metadata.get("author") or ""
        authors = [a.strip() for a in re.split(r"[,;]", authors_meta) if a.strip()]

        doc.close()

        file_hash = compute_file_hash(content)
        metadata_dict = {
            "total_pages": total_pages,
            "authors": authors,
            "file_hash": file_hash,
            "extracted_title": extracted_title,
        }

        return extracted_title, pages_data, metadata_dict

    @staticmethod
    def _heuristic_title_extraction(
        pages_data: List[Dict[str, any]],
        pdf_metadata: Dict[str, str],
        filename: str,
    ) -> str:
        """Infer academic paper title from metadata or first page text."""
        # 1. Check embedded PDF metadata title
        meta_title = pdf_metadata.get("title", "").strip()
        if meta_title and len(meta_title) > 5 and not meta_title.lower().endswith(".pdf"):
            # Avoid generic placeholders
            if not any(placeholder in meta_title.lower() for placeholder in ["untitled", "microsoft word", "latex"]):
                return meta_title

        # 2. Extract from first page
        if pages_data:
            first_page_text = pages_data[0]["text"]
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

            # Candidate lines from the top of page 1
            for line in lines[:8]:
                low = line.lower()
                # Skip arxiv, conference, proceedings, or copyright/permission banners
                if any(k in low for k in ["arxiv:", "conference", "proceedings", "permission to", "copyright", "all rights reserved"]):
                    continue
                if len(line) >= 10 and not line.isdigit():
                    return line

        # 3. Fallback to clean filename
        clean_name = Path(filename).stem
        clean_name = re.sub(r"[_\-\.]+", " ", clean_name).strip()
        return clean_name.title() if clean_name else "Untitled Paper"


pdf_service = PDFService()
