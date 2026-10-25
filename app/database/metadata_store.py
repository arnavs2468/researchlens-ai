"""Metadata store for tracking indexed research papers."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from app.config import settings
from app.schemas import PaperMetadata


class MetadataStore:
    """Manages paper metadata persistence in JSON storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.metadata_db_path
        self._cache: Dict[str, PaperMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        meta = PaperMetadata(**item)
                        self._cache[meta.document_id] = meta
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [meta.model_dump() for meta in self._cache.values()]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_paper(self, paper: PaperMetadata) -> None:
        self._cache[paper.document_id] = paper
        self._save()

    def get_paper(self, document_id: str) -> Optional[PaperMetadata]:
        return self._cache.get(document_id)

    def find_by_hash(self, file_hash: str) -> Optional[PaperMetadata]:
        for paper in self._cache.values():
            if paper.file_hash == file_hash:
                return paper
        return None

    def find_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        clean_id = arxiv_id.strip().lower()
        for paper in self._cache.values():
            if paper.arxiv_id and paper.arxiv_id.strip().lower() == clean_id:
                return paper
        return None

    def list_papers(self) -> List[PaperMetadata]:
        return sorted(
            list(self._cache.values()),
            key=lambda p: p.created_at,
            reverse=True,
        )

    def delete_paper(self, document_id: str) -> bool:
        if document_id in self._cache:
            del self._cache[document_id]
            self._save()
            return True
        return False

    def count_papers(self) -> int:
        return len(self._cache)

    def count_total_chunks(self) -> int:
        return sum(paper.total_chunks for paper in self._cache.values())


metadata_store = MetadataStore()
