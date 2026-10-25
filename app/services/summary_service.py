"""Structured single-paper summarization service."""

import logging
import re
from typing import List, Optional
import groq
from app.config import settings
from app.database.metadata_store import metadata_store
from app.prompts.summary_prompt import SYSTEM_PROMPT_SUMMARY, USER_PROMPT_SUMMARY
from app.schemas import (
    CitationReference,
    RetrievedChunk,
    SummaryRequest,
    SummaryResponse,
    SummarySection,
)
from app.services.retrieval_service import retrieval_service
from app.utils.citations import format_evidence_block, validate_and_build_citations

logger = logging.getLogger(__name__)

STANDARD_SECTIONS = [
    "Research Problem",
    "Motivation",
    "Proposed Method",
    "Dataset & Benchmarks",
    "Experimental Setup",
    "Main Results",
    "Limitations",
    "Future Directions",
]


class SummaryService:
    """Generates structured 8-section paper summaries with citations."""

    def summarize_paper(self, request: SummaryRequest) -> SummaryResponse:
        paper = metadata_store.get_paper(request.document_id)
        if not paper:
            raise ValueError(f"Paper with document_id '{request.document_id}' not found.")

        # Retrieve representative chunks across abstract, introduction, methods, results, conclusion
        section_queries = [
            "abstract research problem motivation",
            "proposed method architecture algorithm",
            "dataset benchmarks experimental setup",
            "results findings performance comparison",
            "limitations future work conclusion",
        ]

        collected_chunks: List[RetrievedChunk] = []
        for q in section_queries:
            chunks = retrieval_service.retrieve(
                query=q,
                retrieval_mode="hybrid",
                top_k=2,
                document_ids=[request.document_id],
            )
            collected_chunks.extend(chunks)

        unique_chunks = retrieval_service._deduplicate_chunks(collected_chunks)
        evidence_block, citation_map = format_evidence_block(unique_chunks)

        if not settings.is_groq_configured:
            return self._build_unconfigured_summary(paper, unique_chunks)

        client = groq.Groq(api_key=settings.groq_api_key)
        user_prompt = USER_PROMPT_SUMMARY.format(
            paper_title=paper.paper_title,
            evidence_block=evidence_block,
        )

        model_name = settings.groq_model
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name,
                temperature=0.1,
                max_tokens=2000,
            )
            raw_text = completion.choices[0].message.content or ""
        except Exception as e:
            return self._build_error_summary(paper, unique_chunks, error=str(e))

        # Parse sections and validate citations
        sections = self._parse_generated_sections(raw_text, citation_map)
        all_citations: List[CitationReference] = []
        for sec in sections:
            all_citations.extend(sec.citations)

        markdown_content = self._build_markdown(paper, sections, all_citations)

        return SummaryResponse(
            document_id=paper.document_id,
            paper_title=paper.paper_title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            sections=sections,
            markdown_content=markdown_content,
            model_used=model_name,
        )

    def _parse_generated_sections(
        self,
        raw_text: str,
        citation_map: dict,
    ) -> List[SummarySection]:
        """Parse numbered or header sections from LLM response."""
        sections: List[SummarySection] = []

        # Regex to split on numbered headers like "1. Research Problem" or "### 1. Research Problem"
        pattern = r"(?:^|\n)(?:#+\s*)?(?:\d+\.?\s*)?([A-Za-z &]+):\s*\n?"
        parts = re.split(pattern, raw_text)

        if len(parts) > 2:
            # Alternate title and content
            for i in range(1, len(parts), 2):
                title = parts[i].strip()
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                val_citations, _ = validate_and_build_citations(content, citation_map)
                sections.append(
                    SummarySection(
                        title=title,
                        content=content,
                        citations=val_citations,
                    )
                )
        else:
            # Fallback: treat whole text as general summary
            val_citations, _ = validate_and_build_citations(raw_text, citation_map)
            sections.append(
                SummarySection(
                    title="Executive Summary",
                    content=raw_text,
                    citations=val_citations,
                )
            )

        return sections

    def _build_markdown(
        self,
        paper,
        sections: List[SummarySection],
        citations: List[CitationReference],
    ) -> str:
        """Create exportable Markdown document."""
        lines = [
            f"# Research Summary: {paper.paper_title}",
            f"**Authors:** {', '.join(paper.authors) if paper.authors else 'N/A'}",
            f"**Source:** {paper.arxiv_id if paper.arxiv_id else paper.source_filename}",
            f"**Indexed Pages:** {paper.total_pages} | **Total Chunks:** {paper.total_chunks}",
            "\n---\n",
        ]

        for sec in sections:
            lines.append(f"## {sec.title}")
            lines.append(f"{sec.content}\n")

        if citations:
            lines.append("## References & Evidence Citations")
            for cit in citations:
                lines.append(
                    f"- **[{cit.citation_id}]** Page {cit.page_number} ({cit.paper_title}): *\"{cit.excerpt}\"*"
                )

        return "\n".join(lines)

    def _build_unconfigured_summary(self, paper, chunks: List[RetrievedChunk]) -> SummaryResponse:
        sections = [
            SummarySection(
                title=sec_name,
                content=(
                    f"Evidence chunks retrieved ({len(chunks)} chunks available). "
                    "Configure `GROQ_API_KEY` in `.env` to generate grounded section synthesis."
                ),
                citations=[],
            )
            for sec_name in STANDARD_SECTIONS
        ]
        return SummaryResponse(
            document_id=paper.document_id,
            paper_title=paper.paper_title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            sections=sections,
            markdown_content=f"# Summary: {paper.paper_title}\n\n*Configure GROQ_API_KEY to generate full AI summary.*",
            model_used="None (GROQ_API_KEY missing)",
            warning="GROQ_API_KEY is not configured.",
        )

    def _build_error_summary(self, paper, chunks, error: str) -> SummaryResponse:
        sections = [
            SummarySection(
                title="Error Generating Summary",
                content=f"An error occurred while calling the Groq API: {error}",
                citations=[],
            )
        ]
        return SummaryResponse(
            document_id=paper.document_id,
            paper_title=paper.paper_title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            sections=sections,
            markdown_content=f"# Error Generating Summary\n\n{error}",
            model_used="Error",
            warning=error,
        )


summary_service = SummaryService()
