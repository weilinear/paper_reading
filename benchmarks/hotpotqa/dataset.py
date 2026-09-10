"""
HotpotQA Dataset Loader and Official Evaluator.
Supports both standard official HotpotQA validation sets and offline mini fixtures.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.base import Benchmark, Metric, Task


class HotpotQABenchmark(Benchmark):
    name = "HotpotQA"

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, split: str = "dev", limit: Optional[int] = None) -> List[Task]:
        """
        Load tasks from json file.
        split:
          - 'dev' or 'validation': Official standard HotpotQA validation set (distractor).
          - 'mini': 5-sample local fixture for fast testing without network.
        """
        if split in ("dev", "validation"):
            file_path = self.data_dir / "hotpot_dev.json"
            if not file_path.exists():
                print(f"[HotpotQA] Official {split} split not found locally. Downloading standard samples...")
                self.download_standard_split(split="validation", target_file=file_path, count=limit or 100)
        else:
            file_path = self.data_dir / f"{split}_hotpotqa.json"
            if not file_path.exists():
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
                metadata={
                    "type": item.get("type", "bridge"),
                    "level": item.get("level", "hard"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                break

        return tasks

    def download_standard_split(
        self,
        split: str = "validation",
        target_file: Optional[Path] = None,
        count: int = 100,
    ) -> Path:
        """
        Download official HotpotQA samples via Hugging Face datasets API.
        Zero third-party pip dependencies required.
        """
        out_path = target_file or (self.data_dir / "hotpot_dev.json")
        batch_size = 100
        samples = []

        for offset in range(0, count, batch_size):
            fetch_limit = min(batch_size, count - offset)
            url = f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split={split}&offset={offset}&limit={fetch_limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                print(f"[Warning] Could not fetch batch at offset {offset}: {e}")
                break

            for r in data.get("rows", []):
                row = r["row"]
                passages = {}
                ctx = row.get("context", {})
                titles = ctx.get("title", [])
                sentences_list = ctx.get("sentences", [])
                for t, sents in zip(titles, sentences_list):
                    passages[t] = " ".join(sents)

                sample = {
                    "id": row["id"],
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "type": row.get("type", "bridge"),
                    "level": row.get("level", "hard"),
                    "supporting_facts": row.get("supporting_facts", []),
                    "passages": passages,
                }
                samples.append(sample)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"[HotpotQA] Saved {len(samples)} official standard samples to {out_path}")
        return out_path

    def evaluate_task(self, task: Task, prediction: str) -> Dict[str, float]:
        """Calculate Exact Match and token-level F1 score using official normalization."""
        em = float(Metric.exact_match(prediction, task.gold_answer))
        f1 = float(Metric.f1_score(prediction, task.gold_answer))
        return {
            "em": em,
            "f1": f1,
        }
