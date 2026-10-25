"""CLI script to run retrieval evaluation and print comparative IR metrics."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.schemas import EvaluationRunRequest
from app.services.evaluation_service import evaluation_service

if __name__ == "__main__":
    print("=" * 70)
    print(" ResearchLens AI — Retrieval Evaluation Benchmark")
    print("=" * 70)

    request = EvaluationRunRequest(
        retrieval_modes=["dense", "bm25", "hybrid"],
        k_values=[3, 5, 10],
    )

    print("Running evaluation across [dense, bm25, hybrid] for K in [3, 5, 10]...")
    results = evaluation_service.run_evaluation(request)

    print(f"\nCompleted evaluation across {results.benchmark_size} benchmark questions.")
    print("-" * 70)
    print(f"{'Mode':<12} {'K':<4} {'Hit Rate':<12} {'Recall':<12} {'MRR':<10} {'Precision':<10}")
    print("-" * 70)

    for m in results.metrics:
        print(
            f"{m.retrieval_mode:<12} {m.k:<4} {m.hit_rate*100:>7.2f}%     "
            f"{m.recall*100:>7.2f}%     {m.mrr:>7.3f}    {m.precision*100:>7.2f}%"
        )

    print("=" * 70)
    print(f"Results saved to: {evaluation_service.results_path}")
