"""
Base classes and standard metrics for agent benchmarks.
Designed to be lightweight, modular, and reusable across different papers and models.
"""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==============================================================================
# Task Representation
# ==============================================================================

@dataclass
class Task:
    """A single evaluation instance in a benchmark."""
    id: str
    question: str
    gold_answer: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Standard Evaluation Metrics (SQuAD / HotpotQA Official Norms)
# ==============================================================================

class Metric:
    """Standard evaluation metrics for question answering and agent outputs."""

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match(cls, prediction: str, ground_truth: str) -> bool:
        """Compute exact match accuracy between prediction and ground truth."""
        return cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth)

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score between prediction and ground truth."""
        normalized_prediction = cls.normalize_answer(prediction)
        normalized_ground_truth = cls.normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            # If either is no-answer, but both are empty -> 1.0
            return int(prediction_tokens == ground_truth_tokens)

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


# ==============================================================================
# Abstract Benchmark Interface
# ==============================================================================

class Benchmark(ABC):
    """Abstract base class for reusable evaluation suites."""

    name: str

    @abstractmethod
    def load_data(self, split: str = "dev", limit: Optional[int] = None) -> List[Task]:
        """Load evaluation tasks for a given split (train, val, dev, test)."""
        pass

    @abstractmethod
    def evaluate_task(self, task: Task, prediction: str) -> Dict[str, float]:
        """Compute metrics for a single task."""
        pass
