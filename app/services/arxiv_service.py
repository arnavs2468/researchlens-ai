"""ArXiv API search and paper download service."""

import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from app.database.metadata_store import metadata_store
from app.schemas import ArxivPaperResult


class ArxivServiceError(Exception):
    """Exception raised for ArXiv query or download issues."""
    pass


class ArxivService:
    """Interfaces with the arXiv API to discover and fetch research papers."""

    BASE_URL = "https://export.arxiv.org/api/query"
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    @classmethod
    async def search_papers(
        cls,
        query: str,
        max_results: int = 10,
    ) -> List[ArxivPaperResult]:
        """
        Search arXiv by topic, title, author, or keywords.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # Build query parameters
        params = {
            "search_query": f"all:{clean_query}",
            "start": 0,
            "max_results": max(1, min(max_results, 30)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        url = f"{cls.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    raise ArxivServiceError(
                        f"ArXiv query failed with HTTP {response.status_code}: {response.text[:200]}"
                    )
                content = response.text
        except httpx.RequestError as e:
            raise ArxivServiceError(f"Network error querying ArXiv: {str(e)}")

        return cls._parse_atom_feed(content)

    @classmethod
    def _parse_atom_feed(cls, xml_text: str) -> List[ArxivPaperResult]:
        """Parse XML feed from arXiv API into ArxivPaperResult objects."""
        results: List[ArxivPaperResult] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ArxivServiceError(f"Error parsing ArXiv XML response: {str(e)}")

        for entry in root.findall("atom:entry", cls.ATOM_NS):
            # ArXiv ID is in entry id URL: http://arxiv.org/abs/2301.12345v1
            entry_id_el = entry.find("atom:id", cls.ATOM_NS)
            raw_id_url = entry_id_el.text.strip() if entry_id_el is not None and entry_id_el.text else ""
            arxiv_id_match = re.search(r"arxiv\.org/abs/([0-9\.]+v?\d*)", raw_id_url)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else raw_id_url.split("/")[-1]

            title_el = entry.find("atom:title", cls.ATOM_NS)
            title = re.sub(r"\s+", " ", title_el.text).strip() if title_el is not None and title_el.text else "Untitled"

            summary_el = entry.find("atom:summary", cls.ATOM_NS)
            summary = re.sub(r"\s+", " ", summary_el.text).strip() if summary_el is not None and summary_el.text else ""

            published_el = entry.find("atom:published", cls.ATOM_NS)
            published = published_el.text.strip() if published_el is not None and published_el.text else ""

            updated_el = entry.find("atom:updated", cls.ATOM_NS)
            updated = updated_el.text.strip() if updated_el is not None and updated_el.text else None

            authors = [
                author.find("atom:name", cls.ATOM_NS).text.strip()
                for author in entry.findall("atom:author", cls.ATOM_NS)
                if author.find("atom:name", cls.ATOM_NS) is not None
                and author.find("atom:name", cls.ATOM_NS).text
            ]

            # Find PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            for link in entry.findall("atom:link", cls.ATOM_NS):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", pdf_url)
                    break

            entry_url = f"https://arxiv.org/abs/{arxiv_id}"

            # Check if this arXiv paper is already indexed
            is_indexed = metadata_store.find_by_arxiv_id(arxiv_id) is not None

            results.append(
                ArxivPaperResult(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=summary,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    entry_url=entry_url,
                    is_already_indexed=is_indexed,
                )
            )

        return results

    @classmethod
    async def download_paper_pdf(cls, arxiv_id: str, pdf_url: Optional[str] = None) -> bytes:
        """
        Download PDF bytes from arXiv.
        """
        url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        headers = {
            "User-Agent": "ResearchLensAI/1.0 (academic research tool; mailto:researchlens@example.edu)"
        }

        try:
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    raise ArxivServiceError(
                        f"Failed to download PDF from {url}. Status code: {response.status_code}"
                    )
                content = response.content
                if not content.startswith(b"%PDF-"):
                    raise ArxivServiceError(
                        f"ArXiv response for {arxiv_id} was not a valid PDF file."
                    )
                return content
        except httpx.RequestError as e:
            raise ArxivServiceError(f"Network error while downloading arXiv paper {arxiv_id}: {str(e)}")


arxiv_service = ArxivService()
