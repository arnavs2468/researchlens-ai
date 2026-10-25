"""Retrieval Evaluation Service: Hit Rate, Recall, MRR, and Precision@K benchmarking."""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings
from app.database.vector_store import vector_store
from app.schemas import (
    BenchmarkQuestion,
    EvaluationRunRequest,
    EvaluationRunResponse,
    MetricResult,
    QuestionEvaluationDetail,
    RetrievedChunk,
)
from app.services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)


class EvaluationService:
    """Computes reproducible retrieval metrics across dense, BM25, and hybrid search."""

    def __init__(self, benchmark_path: Optional[Path] = None):
        self.benchmark_path = benchmark_path or (settings.evaluation_dir / "benchmark_questions.json")
        self.results_path = settings.evaluation_dir / "benchmark_results.json"
        self._questions: List[BenchmarkQuestion] = []
        self.load_benchmark_questions()

    def load_benchmark_questions(self) -> List[BenchmarkQuestion]:
        """Load benchmark questions from storage."""
        if not self.benchmark_path.exists():
            self._questions = []
            return self._questions

        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._questions = [BenchmarkQuestion(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load benchmark questions: {e}")
            self._questions = []

        return self._questions

    def get_benchmark_questions(
        self, category: Optional[str] = None
    ) -> List[BenchmarkQuestion]:
        """Return benchmark questions, optionally filtered by category."""
        if not self._questions:
            self.load_benchmark_questions()

        if category:
            return [q for q in self._questions if q.category.lower() == category.lower()]
        return self._questions

    def add_benchmark_question(self, question: BenchmarkQuestion) -> BenchmarkQuestion:
        """Add or update a question in the benchmark dataset."""
        existing = [q for q in self._questions if q.id != question.id]
        existing.append(question)
        self._questions = existing

        # Persist to disk
        self.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.benchmark_path, "w", encoding="utf-8") as f:
            json.dump([q.model_dump() for q in self._questions], f, indent=2)

        return question

    def run_evaluation(self, request: EvaluationRunRequest) -> EvaluationRunResponse:
        """
        Execute full benchmark evaluation:
        Computes Hit Rate@K, Recall@K, MRR@K, and Precision@K for each retrieval mode and K.
        """
        questions = self.get_benchmark_questions()
        if request.selected_categories:
            cats = {c.lower() for c in request.selected_categories}
            questions = [q for q in questions if q.category.lower() in cats]

        # Filter out unanswerable questions for standard retrieval recall
        evaluable_questions = [q for q in questions if not q.is_unanswerable]

        retrieval_modes = request.retrieval_modes or ["dense", "bm25", "hybrid"]
        k_values = request.k_values or [3, 5, 10]

        all_metric_results: List[MetricResult] = []
        all_details: List[QuestionEvaluationDetail] = []

        total_questions = len(evaluable_questions)
        indexed_chunks_count = vector_store.count_chunks()

        notes = (
            f"Evaluated against {indexed_chunks_count} indexed chunks across {total_questions} "
            "benchmark questions. Metrics reflect ground-truth target paper and keyword relevance."
        )

        for mode in retrieval_modes:
            for k in k_values:
                hits = 0
                reciprocal_rank_sum = 0.0
                recall_sum = 0.0
                precision_sum = 0.0

                for q in evaluable_questions:
                    # Retrieve top-k chunks
                    retrieved = retrieval_service.retrieve(
                        query=q.question,
                        retrieval_mode=mode,
                        top_k=k,
                    )

                    is_hit, rr, rec, prec, ret_doc_ids = self._evaluate_single_query(q, retrieved, k)

                    if is_hit:
                        hits += 1
                    reciprocal_rank_sum += rr
                    recall_sum += rec
                    precision_sum += prec

                    # Keep details for the primary baseline (e.g., hybrid at top k=5)
                    if mode == "hybrid" and k == 5:
                        all_details.append(
                            QuestionEvaluationDetail(
                                question_id=q.id,
                                question=q.question,
                                category=q.category,
                                target_document_ids=q.target_document_ids,
                                retrieved_chunk_doc_ids=ret_doc_ids,
                                hit=is_hit,
                                reciprocal_rank=round(rr, 3),
                                precision_at_k=round(prec, 3),
                            )
                        )

                denom = total_questions if total_questions > 0 else 1
                metric = MetricResult(
                    retrieval_mode=mode,
                    k=k,
                    hit_rate=round(hits / denom, 4),
                    recall=round(recall_sum / denom, 4),
                    mrr=round(reciprocal_rank_sum / denom, 4),
                    precision=round(precision_sum / denom, 4),
                    total_evaluated=total_questions,
                )
                all_metric_results.append(metric)

        run_response = EvaluationRunResponse(
            timestamp=datetime.utcnow().isoformat(),
            benchmark_size=total_questions,
            metrics=all_metric_results,
            details=all_details,
            notes=notes,
        )

        # Persist results
        self._save_results(run_response)
        return run_response

    def _evaluate_single_query(
        self,
        question: BenchmarkQuestion,
        retrieved_chunks: List[RetrievedChunk],
        k: int,
    ) -> tuple[bool, float, float, float, List[str]]:
        """
        Evaluate relevance of retrieved chunks for a single benchmark question.
        Returns (is_hit, reciprocal_rank, recall, precision, list_of_retrieved_doc_ids)
        """
        if not retrieved_chunks:
            return False, 0.0, 0.0, 0.0, []

        ret_doc_ids = [c.document_id for c in retrieved_chunks]
        target_ids = {t.lower() for t in question.target_document_ids}
        keywords = [kw.lower() for kw in question.relevant_keywords]

        relevant_indices: List[int] = []

        for idx, chunk in enumerate(retrieved_chunks):
            # 1. Match document_id or filename
            doc_id_clean = chunk.document_id.lower()
            src_file_clean = chunk.source_filename.lower()
            title_clean = chunk.paper_title.lower()

            matches_doc = any(
                tid in doc_id_clean or tid in src_file_clean or tid in title_clean
                for tid in target_ids
            )

            # 2. Match key semantic facts/keywords in the passage
            matches_content = False
            if keywords:
                text_lower = chunk.text.lower()
                matches_count = sum(1 for kw in keywords if kw in text_lower)
                if matches_count >= max(1, len(keywords) // 2):
                    matches_content = True

            if matches_doc or matches_content:
                relevant_indices.append(idx)

        is_hit = len(relevant_indices) > 0
        rr = (1.0 / (relevant_indices[0] + 1)) if relevant_indices else 0.0
        prec = len(relevant_indices) / float(k)
        # Recall: bounded to [0, 1]
        rec = 1.0 if is_hit else 0.0

        return is_hit, rr, rec, prec, ret_doc_ids

    def _save_results(self, response: EvaluationRunResponse) -> None:
        """Append evaluation result to benchmark_results.json."""
        try:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            existing_runs = []
            if self.results_path.exists():
                with open(self.results_path, "r", encoding="utf-8") as f:
                    existing_runs = json.load(f)
            existing_runs.append(response.model_dump())
            # Keep last 10 runs
            existing_runs = existing_runs[-10:]
            with open(self.results_path, "w", encoding="utf-8") as f:
                json.dump(existing_runs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evaluation run: {e}")

    def get_latest_results(self) -> Optional[EvaluationRunResponse]:
        """Fetch the most recent evaluation run results."""
        if not self.results_path.exists():
            return None
        try:
            with open(self.results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and isinstance(data, list):
                    return EvaluationRunResponse(**data[-1])
        except Exception:
            return None
        return None


evaluation_service = EvaluationService()
