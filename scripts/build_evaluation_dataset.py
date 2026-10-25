"""Script to verify and build the benchmark evaluation dataset."""

import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.services.evaluation_service import evaluation_service

if __name__ == "__main__":
    questions = evaluation_service.get_benchmark_questions()
    print(f"Loaded {len(questions)} evaluation questions from {evaluation_service.benchmark_path}.")

    categories = {}
    for q in questions:
        categories[q.category] = categories.get(q.category, 0) + 1

    print("\nBenchmark Breakdown by Category:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} questions")

    print("\nBenchmark is ready for evaluation.")
