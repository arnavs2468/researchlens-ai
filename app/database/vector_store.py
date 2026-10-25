"""ChromaDB persistent vector store for research paper passages."""

from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.schemas import Chunk, RetrievedChunk
from app.services.embedding_service import embedding_service


class VectorStore:
    """Manages persistent ChromaDB vector storage."""

    COLLECTION_NAME = "research_papers"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or settings.vector_db_dir)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add chunks and embeddings into ChromaDB."""
        if not chunks:
            return

        if embeddings is None:
            texts = [c.text for c in chunks]
            embeddings = embedding_service.embed_texts(texts)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.metadata.document_id,
                "paper_title": c.metadata.paper_title,
                "authors": ", ".join(c.metadata.authors),
                "arxiv_id": c.metadata.arxiv_id or "",
                "source_filename": c.metadata.source_filename,
                "page_number": c.metadata.page_number,
                "chunk_index": c.metadata.chunk_index,
                "section": c.metadata.section or "Body",
                "publication_date": c.metadata.publication_date or "",
            }
            for c in chunks
        ]

        # Insert in batches of 100 to avoid ChromaDB batch limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def delete_paper_chunks(self, document_id: str) -> None:
        """Remove all chunks associated with a document_id."""
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass

    def query_dense(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Perform dense vector retrieval using cosine similarity."""
        count = self.collection.count()
        if count == 0:
            return []

        query_emb = embedding_service.embed_query(query)
        actual_k = min(top_k, count)

        where_filter: Optional[Dict[str, Any]] = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            # For cosine distance in chromadb, similarity score = 1 - distance
            score = round(max(0.0, 1.0 - float(dist)), 4)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", ""),
                    paper_title=meta.get("paper_title", "Untitled"),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    section=meta.get("section") or None,
                    arxiv_id=meta.get("arxiv_id") or None,
                    source_filename=meta.get("source_filename", ""),
                    text=text,
                    score=score,
                    retrieval_method="dense",
                )
            )

        return retrieved

    def get_all_chunks(self, document_ids: Optional[List[str]] = None) -> List[RetrievedChunk]:
        """Retrieve chunks from ChromaDB for lexical indexing or full-paper extraction."""
        count = self.collection.count()
        if count == 0:
            return []

        where_filter: Optional[Dict[str, Any]] = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        # Fetch in pages
        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        all_chunks: List[RetrievedChunk] = []
        if not results or not results["ids"]:
            return []

        for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            all_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", ""),
                    paper_title=meta.get("paper_title", "Untitled"),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    section=meta.get("section") or None,
                    arxiv_id=meta.get("arxiv_id") or None,
                    source_filename=meta.get("source_filename", ""),
                    text=doc,
                    score=1.0,
                    retrieval_method="dense",
                )
            )

        return all_chunks

    def count_chunks(self) -> int:
        return self.collection.count()


vector_store = VectorStore()
