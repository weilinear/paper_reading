#!/usr/bin/env python3
"""
Unit tests for HotpotQA benchmark and execution environment.
"""

import unittest
from benchmarks.base import Metric, Task
from benchmarks.hotpotqa import HotpotQABenchmark, HotpotQAEnv


class TestHotpotQABenchmark(unittest.TestCase):
    def test_metrics(self):
        # Exact Match
        self.assertTrue(Metric.exact_match("Chief of Protocol", "chief of protocol"))
        self.assertTrue(Metric.exact_match("The United States", "United States"))
        self.assertFalse(Metric.exact_match("New York City", "Chicago"))

        # F1 score
        self.assertAlmostEqual(Metric.f1_score("Arthur's Magazine", "Arthur's Magazine"), 1.0)
        self.assertAlmostEqual(Metric.f1_score("Arthur's Magazine", "Magazine"), 0.6666667, places=4)
        self.assertEqual(Metric.f1_score("Paris", "London"), 0.0)

    def test_environment_execution(self):
        passages = {
            "Kiss and Tell": "Kiss and Tell is a comedy film starring Shirley Temple as Corliss Archer.",
            "Shirley Temple": "Shirley Temple was an American actress who served as Chief of Protocol.",
        }
        env = HotpotQAEnv(passages)

        # Search Hop 1
        obs, done = env.step("Search[Kiss and Tell]")
        self.assertFalse(done)
        self.assertIn("Shirley Temple", obs)

        # Lookup
        obs, done = env.step("Lookup[Corliss Archer]")
        self.assertFalse(done)
        self.assertIn("Corliss Archer", obs)

        # Search Hop 2
        obs, done = env.step("Search[Shirley Temple]")
        self.assertFalse(done)
        self.assertIn("Chief of Protocol", obs)

        # Finish
        obs, done = env.step("Finish[Chief of Protocol]")
        self.assertTrue(done)
        self.assertEqual(env.final_answer, "Chief of Protocol")

    def test_dataset_loader(self):
        bm = HotpotQABenchmark()
        tasks = bm.load_data(split="mini")
        self.assertTrue(len(tasks) >= 5)
        t0 = tasks[0]
        self.assertIsInstance(t0, Task)
        self.assertIn("Kiss and Tell", t0.question)
        self.assertEqual(t0.gold_answer, "Chief of Protocol of the United States")


if __name__ == "__main__":
    unittest.main()
