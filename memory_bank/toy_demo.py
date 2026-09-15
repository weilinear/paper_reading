#!/usr/bin/env python3
"""
memory_bank/toy_demo.py - Interactive walkthrough demonstrating MemoryBank.

Demonstrates the 3 core pillars from Zhong et al. (AAAI 2024):
1. Multi-layered Storage (User Portrait, Event Summaries, Dialogue Log).
2. Ebbinghaus Forgetting Curve & Spacing-Effect Reinforcement.
3. Memory-Augmented Prompt Assembly for an AI Companion (SiliconFriend).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_bank.storage import MemoryBank


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def print_retention_table(bank: MemoryBank, current_day: float) -> None:
    print(f"\n[Memory Status on Day {current_day:.1f}] (Threshold = {bank.forgetting_threshold:.2f})")
    print("-" * 70)
    print(f"{'ID':<8} | {'Content Excerpt':<30} | {'Strength S':<10} | {'Elapsed t':<9} | {'Retention R':<11} | {'Status'}")
    print("-" * 70)
    for m in bank.memories:
        elapsed = current_day - m.last_accessed_day
        r = m.get_retention(current_day)
        status = "RETAINED" if r >= bank.forgetting_threshold else "FORGOTTEN"
        excerpt = (m.content[:28] + "..") if len(m.content) > 28 else m.content
        print(
            f"{m.id:<8} | {excerpt:<30} | {m.strength:<10.1f} | {elapsed:<9.1f} | {r:<11.4f} | {status}"
        )
    print("-" * 70)


def main():
    print_banner("MemoryBank Walkthrough: Enhancing LLMs with Long-Term Memory")
    print("Inspired by Ebbinghaus Forgetting Curve: R = exp(-t / S)")
    print("• t = elapsed time since last recall")
    print("• S = memory strength (reinforced when recalled: S <- S + 1, t <- 0)")

    # Initialize MemoryBank
    # For this toy demo, forgetting threshold is set to 0.15 (15% retention)
    bank = MemoryBank(forgetting_threshold=0.15, user_name="Linda")

    # --------------------------------------------------------------------------
    # DAY 1: Initial Interactions
    # --------------------------------------------------------------------------
    print_banner("Step 1: Day 1.0 - Initial Conversations & Storage")
    day = 1.0

    print("• Linda and SiliconFriend have two conversations:")
    # Memory 1: Important learning advice
    m1 = bank.record_turn(
        current_day=day,
        user_input="I want to learn Python programming. What book do you suggest?",
        ai_response="I suggest 'Automate the Boring Stuff with Python' by Al Sweigart.",
    )
    print(f"  -> Added {m1.id}: Python book recommendation (Automate the Boring Stuff with Python)")

    # Memory 2: Fleeting casual detail
    m2 = bank.record_turn(
        current_day=day,
        user_input="I had a quick ham sandwich for lunch today.",
        ai_response="Sounds tasty! Remember to take a good break from work.",
    )
    print(f"  -> Added {m2.id}: Casual lunch comment (ham sandwich)")

    # Memory 3: Algorithm discussion
    m3 = bank.record_turn(
        current_day=day,
        user_input="Can you explain how quicksort works?",
        ai_response="Quicksort is a divide-and-conquer algorithm with average O(n log n) complexity.",
    )
    print(f"  -> Added {m3.id}: Technical explanation of quicksort")

    # Update daily event summary
    bank.event_summary.add_daily_summary(
        day=1,
        summary="Linda asked for Python learning resources (book recommended: Automate the Boring Stuff) and quicksort.",
    )

    print_retention_table(bank, current_day=day)

    # --------------------------------------------------------------------------
    # DAY 2: Persona & Psychological Modeling
    # --------------------------------------------------------------------------
    print_banner("Step 2: Day 2.0 - Emotional Interaction & User Portrait Update")
    day = 2.0

    m4 = bank.record_turn(
        current_day=day,
        user_input="I've been feeling stressed at my job lately. I feel like I'm not learning fast enough.",
        ai_response="Learning to code takes patience. You are making steady progress, so give yourself credit.",
    )
    print(f"  -> Added {m4.id}: Emotional check-in on job stress")

    # MemoryBank updates the dynamic User Portrait
    bank.user_portrait.traits = ["introverted", "ambitious", "growth-oriented", "hardworking"]
    bank.user_portrait.interests = ["Python programming", "algorithms", "reading"]
    bank.user_portrait.emotional_state = "experiencing workplace learning anxiety"

    bank.event_summary.add_daily_summary(
        day=2,
        summary="Linda shared feelings of workplace stress and received encouraging guidance on learning pace.",
    )

    print("\nUpdated Global User Portrait:")
    print(bank.user_portrait.format_portrait())

    # --------------------------------------------------------------------------
    # DAY 4: Ebbinghaus Time Decay
    # --------------------------------------------------------------------------
    print_banner("Step 3: Day 4.0 - Time Passes (Elapsed t = 3.0 days)")
    day = 4.0

    print("No interaction for 2 days. Notice how retention R drops exponentially for all memories:")
    print("Formula: R = exp(-t / S) = exp(-3.0 / 1.0) = 0.0498 (< 0.15 threshold!)")
    print_retention_table(bank, current_day=day)

    # --------------------------------------------------------------------------
    # DAY 4.1: Memory Probing Query & Spacing-Effect Reinforcement
    # --------------------------------------------------------------------------
    print_banner("Step 4: Day 4.1 - Memory Retrieval & Reinforcement (The Spacing Effect)")
    query = "What Python book did you recommend to me earlier?"
    print(f"Linda asks: '{query}'")

    # Retrieve with forgetting model disabled momentarily to locate the memory
    retrieved = bank.retrieve(query, current_day=day, top_k=1, apply_forgetting=False)
    if retrieved:
        recalled_mem, sim, old_r = retrieved[0]
        print(f"\n✓ Successfully retrieved memory: [{recalled_mem.id}] (Semantic similarity: {sim:.3f})")
        print(f"  Content: {recalled_mem.content}")
        print(f"\n⚡ SPACING EFFECT TRIGGERED:")
        print(f"  Memory strength S increased from {recalled_mem.strength - 1.0:.1f} to {recalled_mem.strength:.1f}!")
        print(f"  Last accessed day reset to Day {recalled_mem.last_accessed_day:.1f} (elapsed t resets to 0).")
        print(f"  New retention R = exp(-0 / {recalled_mem.strength:.1f}) = 1.0000!")

    # --------------------------------------------------------------------------
    # DAY 7: The Contrast: Reinforced vs Unreinforced Decay
    # --------------------------------------------------------------------------
    print_banner("Step 5: Day 7.0 - The Divergence (Reinforced vs Unreinforced Memories)")
    day = 7.0

    print("Now look at the retention table on Day 7.0 (3 days after the recall):")
    print("• Unreinforced memory mem_002 (lunch sandwich): elapsed t = 6 days, S = 1.0 -> R = exp(-6) = 0.0025 -> FORGOTTEN!")
    print("• Reinforced memory mem_001 (Python book):      elapsed t = 3 days, S = 2.0 -> R = exp(-3/2) = 0.2231 -> STILL RETAINED!")

    print_retention_table(bank, current_day=day)

    # --------------------------------------------------------------------------
    # DAY 7.1: Full Memory-Augmented Prompt Assembly
    # --------------------------------------------------------------------------
    print_banner("Step 6: Day 7.1 - MemoryBank Prompt Assembly for the LLM")
    current_query = "Can you remind me of that Python book? And how should I study it this weekend?"

    prompt = bank.format_prompt(current_query=current_query, current_day=day, top_k=2)
    print(prompt)

    print("\n" + "=" * 70)
    print("Summary of Why MemoryBank Works:")
    print("1. Selective Retention: Trivial details (e.g. sandwiches) fade naturally.")
    print("2. Spacing Reinforcement: Frequently referenced facts endure across weeks.")
    print("3. Context Efficiency: LLM receives only relevant memories + user portrait,")
    print("   preventing context window bloat while maintaining deep personalization.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
