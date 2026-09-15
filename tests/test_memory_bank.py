#!/usr/bin/env python3
"""
Unit tests for MemoryBank storage, Ebbinghaus decay, and retrieval.
"""

import math
import unittest
from memory_bank import EventSummary, MemoryBank, MemoryPiece, UserPortrait


class TestMemoryBank(unittest.TestCase):
    def test_ebbinghaus_decay(self):
        # Day 0: Initial creation
        mem = MemoryPiece(
            id="mem_1",
            content="Favorite food is sushi",
            created_day=0.0,
            last_accessed_day=0.0,
            strength=1.0,
        )
        self.assertAlmostEqual(mem.get_retention(current_day=0.0), 1.0)

        # Day 1: t = 1, S = 1 -> R = exp(-1) ≈ 0.3679
        self.assertAlmostEqual(mem.get_retention(current_day=1.0), math.exp(-1.0), places=4)

        # Day 3: t = 3, S = 1 -> R = exp(-3) ≈ 0.0498
        self.assertAlmostEqual(mem.get_retention(current_day=3.0), math.exp(-3.0), places=4)

    def test_spacing_effect_reinforcement(self):
        mem = MemoryPiece(
            id="mem_2",
            content="Studying Python",
            created_day=0.0,
            last_accessed_day=0.0,
            strength=1.0,
        )
        # Access on Day 2
        mem.reinforce(current_day=2.0)
        self.assertEqual(mem.strength, 2.0)
        self.assertEqual(mem.last_accessed_day, 2.0)
        self.assertEqual(mem.access_count, 2)

        # Immediate retention is reset to 1.0
        self.assertAlmostEqual(mem.get_retention(current_day=2.0), 1.0)

        # Day 4 (2 days after access): t = 2, S = 2 -> R = exp(-2/2) = exp(-1) ≈ 0.3679
        # Decays at half the speed of S=1!
        self.assertAlmostEqual(mem.get_retention(current_day=4.0), math.exp(-1.0), places=4)

    def test_memory_bank_retrieval_and_forgetting(self):
        bank = MemoryBank(forgetting_threshold=0.15, user_name="Alice")
        bank.add_memory("Alice loves classical piano music", current_day=1.0)
        bank.add_memory("Alice ate an apple for snack", current_day=1.0)

        # Day 1: Both memories should be retrievable
        res_day1 = bank.retrieve("piano music", current_day=1.0)
        self.assertTrue(len(res_day1) >= 1)
        self.assertIn("piano", res_day1[0][0].content)

        # Advance to Day 6 (5 days without access for apple)
        # Piano was accessed on Day 1 (reinforcing it to S=2.0, t_access=1.0)
        # On Day 6:
        # Piano: t = 6 - 1 = 5, S = 2.0 -> R = exp(-2.5) ≈ 0.082 (below 0.15 threshold unless reinforced again)
        # Apple: t = 6 - 1 = 5, S = 1.0 -> R = exp(-5) ≈ 0.0067 (deeply forgotten)
        res_apple = bank.retrieve("snack", current_day=6.0, apply_forgetting=True)
        self.assertEqual(len(res_apple), 0, "Unreinforced snack memory should have decayed below threshold")

    def test_prompt_formatting(self):
        bank = MemoryBank(user_name="Bob")
        bank.user_portrait.traits = ["creative", "analytical"]
        bank.event_summary.add_daily_summary(1, "Bob discussed machine learning projects.")
        bank.add_memory("Bob's primary project is computer vision", current_day=1.0)

        prompt = bank.format_prompt("What is my project about?", current_day=1.0)
        self.assertIn("GLOBAL USER PORTRAIT", prompt)
        self.assertIn("creative, analytical", prompt)
        self.assertIn("HIERARCHICAL EVENT SUMMARY", prompt)
        self.assertIn("computer vision", prompt)


if __name__ == "__main__":
    unittest.main()
