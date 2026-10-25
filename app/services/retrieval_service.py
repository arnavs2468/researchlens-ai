"""Retrieval engine supporting Dense Vector, BM25, and Hybrid (RRF) search."""

import re
from typing import Dict, List, Optional, Set
from rank_bm25 import BM25Okapi
from app.config import settings
from app.database.vector_store import vector_store
from app.schemas import RetrievedChunk


class RetrievalService:
    """Manages dense, lexical (BM25), and hybrid RRF retrieval."""

    def __init__(self):
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus_chunks: List[RetrievedChunk] = []
        self._cross_encoder = None

    def refresh_bm25_index(self) -> None:
        """Fetch all chunks from vector store and build an in-memory BM25 index."""
        all_chunks = vector_store.get_all_chunks()
        self._bm25_corpus_chunks = all_chunks

        if all_chunks:
            tokenized_corpus = [self._tokenize(chunk.text) for chunk in all_chunks]
            self._bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_index = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase word tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def retrieve(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        enable_reranking: bool = False,
    ) -> List[RetrievedChunk]:
        """
        Unified retrieval entrypoint supporting 'dense', 'bm25', and 'hybrid'.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        mode = (retrieval_mode or settings.default_retrieval_mode).lower()

        if mode == "dense":
            results = self.retrieve_dense(clean_query, top_k=top_k, document_ids=document_ids)
        elif mode == "bm25":
            results = self.retrieve_bm25(clean_query, top_k=top_k, document_ids=document_ids)
        elif mode == "hybrid":
            results = self.retrieve_hybrid(clean_query, top_k=top_k, document_ids=document_ids)
        else:
            results = self.retrieve_hybrid(clean_query, top_k=top_k, document_ids=document_ids)

        # Deduplicate near-identical chunks
        results = self._deduplicate_chunks(results)

        # Optional Cross-Encoder reranking if requested and enabled
        if (enable_reranking or settings.enable_cross_encoder) and len(results) > 1:
            results = self._rerank_with_cross_encoder(clean_query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Perform dense semantic search with ChromaDB."""
        return vector_store.query_dense(query=query, top_k=top_k, document_ids=document_ids)

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Perform keyword search with BM25."""
        if self._bm25_index is None or len(self._bm25_corpus_chunks) != vector_store.count_chunks():
            self.refresh_bm25_index()

        if self._bm25_index is None or not self._bm25_corpus_chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25_index.get_scores(tokenized_query)

        # Filter by document_ids if provided
        filtered_indices = []
        for idx, chunk in enumerate(self._bm25_corpus_chunks):
            if document_ids and chunk.document_id not in document_ids:
                continue
            if scores[idx] > 0.0:  # Only chunks with positive term match
                filtered_indices.append((idx, scores[idx]))

        filtered_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = filtered_indices[:top_k]

        results: List[RetrievedChunk] = []
        for idx, score in top_indices:
            c = self._bm25_corpus_chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    paper_title=c.paper_title,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    section=c.section,
                    arxiv_id=c.arxiv_id,
                    source_filename=c.source_filename,
                    text=c.text,
                    score=round(float(score), 4),
                    retrieval_method="bm25",
                )
            )

        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Combine Dense Vector and BM25 retrieval using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = sum(1 / (rrf_k + rank_i(d)))
        """
        # Fetch candidate pools from both methods (larger pool for fusion)
        fetch_pool = max(top_k * 3, 15)
        dense_results = self.retrieve_dense(query, top_k=fetch_pool, document_ids=document_ids)
        bm25_results = self.retrieve_bm25(query, top_k=fetch_pool, document_ids=document_ids)

        if not dense_results and not bm25_results:
            return []

        if not bm25_results:
            return dense_results[:top_k]

        if not dense_results:
            return bm25_results[:top_k]

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # 1. Rank dense results
        for rank, chunk in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Rank BM25 results
        for rank, chunk in enumerate(bm25_results, start=1):
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort by combined RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results: List[RetrievedChunk] = []
        for cid in sorted_chunk_ids[:top_k]:
            base_chunk = chunk_map[cid]
            fused_results.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    document_id=base_chunk.document_id,
                    paper_title=base_chunk.paper_title,
                    page_number=base_chunk.page_number,
                    chunk_index=base_chunk.chunk_index,
                    section=base_chunk.section,
                    arxiv_id=base_chunk.arxiv_id,
                    source_filename=base_chunk.source_filename,
                    text=base_chunk.text,
                    score=round(rrf_scores[cid], 5),
                    retrieval_method="hybrid",
                )
            )

        return fused_results

    @staticmethod
    def _deduplicate_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate or near-identical text chunks."""
        seen_texts: Set[str] = set()
        unique: List[RetrievedChunk] = []

        for chunk in chunks:
            # Use normalized first 60 chars as fingerprint
            norm_snippet = re.sub(r"\s+", " ", chunk.text[:60].strip().lower())
            if norm_snippet not in seen_texts:
                seen_texts.add(norm_snippet)
                unique.append(chunk)

        return unique

    def _rerank_with_cross_encoder(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Optional cross-encoder reranking."""
        try:
            if self._cross_encoder is None:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(settings.cross_encoder_model)

            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self._cross_encoder.predict(pairs)

            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            reranked_chunks = []
            for chunk, score in ranked[:top_k]:
                chunk.score = round(float(score), 4)
                reranked_chunks.append(chunk)

            return reranked_chunks
        except Exception:
            # Graceful fallback to un-reranked chunks
            return chunks[:top_k]


retrieval_service = RetrievalService()
