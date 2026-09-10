"""
HotpotQA Dataset Loader and Evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.base import Benchmark, Metric, Task


class HotpotQABenchmark(Benchmark):
    name = "HotpotQA"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"

    def load_data(self, split: str = "mini", limit: Optional[int] = None) -> List[Task]:
        """Load tasks from json file."""
        file_path = self.data_dir / f"{split}_hotpotqa.json"
        if not file_path.exists():
            # Fallback to mini
            file_path = self.data_dir / "mini_hotpotqa.json"

        with open(file_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        tasks = []
        for item in raw_items:
            task = Task(
                id=item["id"],
                question=item["question"],
                gold_answer=item["gold_answer"],
                context=item.get("passages", {}),
                metadata=item,
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break

        return tasks

    def evaluate_task(self, task: Task, prediction: str) -> Dict[str, float]:
        """Calculate Exact Match and token-level F1 score."""
        em = float(Metric.exact_match(prediction, task.gold_answer))
        f1 = float(Metric.f1_score(prediction, task.gold_answer))
        return {
            "em": em,
            "f1": f1,
        }
