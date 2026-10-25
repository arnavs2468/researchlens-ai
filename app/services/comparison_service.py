"""Multi-paper comparative analysis service."""

import json
import logging
from typing import Dict, List, Optional
import groq
from app.config import settings
from app.database.metadata_store import metadata_store
from app.prompts.comparison_prompt import SYSTEM_PROMPT_COMPARISON, USER_PROMPT_COMPARISON
from app.schemas import (
    CitationReference,
    CompareRequest,
    CompareResponse,
    ComparisonRow,
    RetrievedChunk,
)
from app.services.retrieval_service import retrieval_service
from app.utils.citations import format_evidence_block, validate_and_build_citations

logger = logging.getLogger(__name__)

COMPARISON_ASPECTS = [
    "Research Problem",
    "Main Contribution",
    "Methodology",
    "Model Architecture or Algorithm",
    "Dataset and Benchmarks",
    "Evaluation Metrics",
    "Main Results",
    "Limitations",
    "Future Research Directions",
]


class ComparisonService:
    """Orchestrates structured comparison across 2 to 5 research papers."""

    def compare_papers(self, request: CompareRequest) -> CompareResponse:
        """
        Compare 2-5 research papers based on retrieved evidence.
        """
        doc_ids = request.document_ids
        if len(doc_ids) < 2 or len(doc_ids) > 5:
            raise ValueError("Comparison requires between 2 and 5 papers.")

        # Resolve titles
        paper_titles: Dict[str, str] = {}
        for did in doc_ids:
            paper = metadata_store.get_paper(did)
            paper_titles[did] = paper.paper_title if paper else f"Paper ({did})"

        # Retrieve evidence chunks for each paper
        # Target queries covering methodology, experiments, and conclusions
        aspect_queries = [
            "methodology model architecture algorithm",
            "dataset benchmarks evaluation metrics experimental results",
            "limitations future work conclusion",
        ]

        all_collected_chunks: List[RetrievedChunk] = []
        for did in doc_ids:
            for q in aspect_queries:
                chunks = retrieval_service.retrieve(
                    query=q,
                    retrieval_mode="hybrid",
                    top_k=3,
                    document_ids=[did],
                )
                all_collected_chunks.extend(chunks)

        # Deduplicate
        unique_chunks = retrieval_service._deduplicate_chunks(all_collected_chunks)
        evidence_block, citation_map = format_evidence_block(unique_chunks)

        # Build manifest
        manifest_lines = [
            f"- Document ID: {did} | Title: \"{paper_titles[did]}\""
            for did in doc_ids
        ]
        manifest_text = "\n".join(manifest_lines)

        # Check Groq configuration
        if not settings.is_groq_configured:
            return self._build_unconfigured_response(doc_ids, paper_titles, unique_chunks)

        client = groq.Groq(api_key=settings.groq_api_key)
        user_prompt = USER_PROMPT_COMPARISON.format(
            papers_manifest=manifest_text,
            evidence_block=evidence_block,
        )

        model_name = settings.groq_model
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMPARISON},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=2500,
            )
            raw_content = completion.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)
        except Exception as e:
            return self._build_fallback_response(doc_ids, paper_titles, unique_chunks, error=str(e))

        # Parse table rows and citations
        comparison_table: List[ComparisonRow] = []
        all_cited_refs: List[CitationReference] = []

        for row_data in parsed.get("comparison_table", []):
            aspect = row_data.get("aspect", "Aspect")
            values = row_data.get("values", {})

            # Clean and validate citations inside each value
            row_citations: Dict[str, List[CitationReference]] = {}
            for did, val_text in values.items():
                val_citations, _ = validate_and_build_citations(val_text, citation_map)
                row_citations[did] = val_citations
                all_cited_refs.extend(val_citations)

            comparison_table.append(
                ComparisonRow(
                    aspect=aspect,
                    values=values,
                    citations=row_citations,
                )
            )

        synthesis = parsed.get("synthesis", {
            "key_similarities": "Not available.",
            "critical_differences": "Not available.",
            "trade_offs": "Not available.",
            "open_questions": "Not available.",
        })

        return CompareResponse(
            document_ids=doc_ids,
            paper_titles=paper_titles,
            comparison_table=comparison_table,
            synthesis=synthesis,
            citations=all_cited_refs,
            model_used=model_name,
        )

    def _build_unconfigured_response(
        self,
        doc_ids: List[str],
        paper_titles: Dict[str, str],
        chunks: List[RetrievedChunk],
    ) -> CompareResponse:
        """Provide structured table template when Groq key is not configured."""
        rows = []
        for aspect in COMPARISON_ASPECTS:
            rows.append(
                ComparisonRow(
                    aspect=aspect,
                    values={
                        did: f"Evidence retrieved ({len([c for c in chunks if c.document_id == did])} passages). Configure GROQ_API_KEY to generate comparative synthesis."
                        for did in doc_ids
                    },
                    citations={},
                )
            )
        return CompareResponse(
            document_ids=doc_ids,
            paper_titles=paper_titles,
            comparison_table=rows,
            synthesis={
                "key_similarities": "Configure GROQ_API_KEY to generate automated cross-paper similarity analysis.",
                "critical_differences": "Configure GROQ_API_KEY to generate automated cross-paper contrast analysis.",
                "trade_offs": "Configure GROQ_API_KEY to analyze model and computational trade-offs.",
                "open_questions": "Configure GROQ_API_KEY to synthesize open research directions.",
            },
            citations=[],
            model_used="None (GROQ_API_KEY not configured)",
            warning="GROQ_API_KEY is not configured. Displaying evidence scaffold.",
        )

    def _build_fallback_response(
        self,
        doc_ids: List[str],
        paper_titles: Dict[str, str],
        chunks: List[RetrievedChunk],
        error: str,
    ) -> CompareResponse:
        """Provide fallback response if Groq API encounters an error."""
        rows = [
            ComparisonRow(
                aspect=aspect,
                values={did: "Analysis interrupted by API error." for did in doc_ids},
                citations={},
            )
            for aspect in COMPARISON_ASPECTS
        ]
        return CompareResponse(
            document_ids=doc_ids,
            paper_titles=paper_titles,
            comparison_table=rows,
            synthesis={
                "key_similarities": f"Error during synthesis: {error}",
                "critical_differences": "Unavailable",
                "trade_offs": "Unavailable",
                "open_questions": "Unavailable",
            },
            citations=[],
            model_used="Error",
            warning=f"Comparison generation failed: {error}",
        )


comparison_service = ComparisonService()
