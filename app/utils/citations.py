"""Citation processing, formatting, and validation utilities."""

import re
from typing import Dict, List, Set, Tuple
from app.schemas import CitationReference, RetrievedChunk


def format_evidence_block(chunks: List[RetrievedChunk]) -> Tuple[str, Dict[str, RetrievedChunk]]:
    """
    Format retrieved chunks into a structured evidence string with stable source IDs [S1], [S2]...
    Returns the formatted text and a mapping from citation_id to RetrievedChunk.
    """
    evidence_lines: List[str] = []
    citation_map: Dict[str, RetrievedChunk] = {}

    for idx, chunk in enumerate(chunks, start=1):
        citation_id = f"S{idx}"
        citation_map[citation_id] = chunk

        section_str = f" | Section: {chunk.section}" if chunk.section else ""
        header = f"[{citation_id}] Paper: \"{chunk.paper_title}\" | Page: {chunk.page_number}{section_str}"
        evidence_lines.append(f"{header}\nPassage:\n{chunk.text.strip()}\n")

    evidence_text = "\n---\n".join(evidence_lines)
    return evidence_text, citation_map


def extract_citations_from_text(text: str) -> List[str]:
    """
    Extract citation identifiers like S1, S2 from text containing [S1], [S2], [S1, S2], etc.
    """
    # Match patterns like [S1], [S2], [S1, S2], [S1,S2], [S1][S2]
    pattern = r"\[\s*S(\d+)(?:\s*,\s*S(\d+))*\s*\]"
    found_ids: Set[str] = set()

    # Search for all [S\d+] individual occurrences
    single_pattern = r"\[\s*(S\d+)\s*\]"
    for match in re.finditer(single_pattern, text):
        found_ids.add(match.group(1))

    # Also handle comma-separated brackets like [S1, S2, S3]
    bracket_pattern = r"\[([^\]]+)\]"
    for match in re.finditer(bracket_pattern, text):
        content = match.group(1)
        sub_ids = re.findall(r"\bS\d+\b", content)
        for s in sub_ids:
            found_ids.add(s)

    # Sort numerically by citation number (S1, S2, S10)
    sorted_citations = sorted(
        list(found_ids),
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999
    )
    return sorted_citations


def validate_and_build_citations(
    generated_text: str,
    citation_map: Dict[str, RetrievedChunk],
    max_excerpt_len: int = 280,
) -> Tuple[List[CitationReference], List[str]]:
    """
    Verify citations found in the text against the allowed citation_map.
    Returns:
        (validated_citations, unvalidated_citation_ids)
    """
    cited_ids = extract_citations_from_text(generated_text)
    validated: List[CitationReference] = []
    unvalidated: List[str] = []

    for cid in cited_ids:
        if cid in citation_map:
            chunk = citation_map[cid]
            # Create a clean excerpt
            raw_text = chunk.text.strip()
            if len(raw_text) > max_excerpt_len:
                excerpt = raw_text[:max_excerpt_len].rstrip() + "..."
            else:
                excerpt = raw_text

            arxiv_url = (
                f"https://arxiv.org/abs/{chunk.arxiv_id}"
                if chunk.arxiv_id
                else None
            )

            validated.append(
                CitationReference(
                    citation_id=cid,
                    document_id=chunk.document_id,
                    paper_title=chunk.paper_title,
                    page_number=chunk.page_number,
                    excerpt=excerpt,
                    arxiv_id=chunk.arxiv_id,
                    source_filename=chunk.source_filename,
                    paper_url=arxiv_url,
                    is_validated=True,
                    section=chunk.section,
                )
            )
        else:
            unvalidated.append(cid)

    return validated, unvalidated
