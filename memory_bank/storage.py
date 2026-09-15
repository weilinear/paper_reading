"""
memory_bank/storage.py - Core implementation of the MemoryBank mechanism.
Implements the 3 pillars:
1. Multi-layered Storage (Dialogue log, Hierarchical Event Summary, User Portrait).
2. Dual-tower Retrieval (Semantic vector matching with Cosine Similarity).
3. Ebbinghaus Forgetting Curve and Spacing-Effect Memory Updating.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemoryPiece:
    """A single atomic memory piece in the MemoryBank."""
    id: str
    content: str
    created_day: float
    last_accessed_day: float
    strength: float = 1.0          # S in Ebbinghaus formula: R = exp(-t / S)
    access_count: int = 1
    category: str = "dialogue"     # 'dialogue', 'event', 'preference'

    def get_retention(self, current_day: float) -> float:
        """
        Calculate memory retention R based on Ebbinghaus Forgetting Curve:
        R = exp(-t / S)
        where:
          t = current_day - last_accessed_day (elapsed time since last recall)
          S = memory strength (increases with repetition)
        """
        elapsed = max(0.0, current_day - self.last_accessed_day)
        # S must be > 0
        s = max(0.1, self.strength)
        return math.exp(-elapsed / s)

    def reinforce(self, current_day: float) -> None:
        """
        Spacing Effect: When a memory is successfully recalled, its memory strength S
        increases and the elapsed time resets to 0, making subsequent forgetting slower.
        """
        self.strength += 1.0
        self.last_accessed_day = current_day
        self.access_count += 1


@dataclass
class UserPortrait:
    """Dynamic profile of user personality, interests, and emotional state."""
    name: str = "User"
    traits: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    emotional_state: str = "neutral"

    def format_portrait(self) -> str:
        parts = [f"Name: {self.name}"]
        if self.traits:
            parts.append(f"Personality Traits: {', '.join(self.traits)}")
        if self.interests:
            parts.append(f"Interests & Hobbies: {', '.join(self.interests)}")
        if self.emotional_state:
            parts.append(f"Current Emotional State: {self.emotional_state}")
        return "\n".join(parts)


@dataclass
class EventSummary:
    """Hierarchical summary of past interactions."""
    daily_summaries: Dict[int, str] = field(default_factory=dict)
    global_summary: str = ""

    def add_daily_summary(self, day: int, summary: str) -> None:
        self.daily_summaries[day] = summary
        # Update global summary
        all_daily = [f"Day {d}: {s}" for d, s in sorted(self.daily_summaries.items())]
        self.global_summary = " | ".join(all_daily)


# ==============================================================================
# Lightweight Semantic Vector Retriever (Pure Python)
# ==============================================================================

class SimpleVectorRetriever:
    """
    Lightweight term-frequency vector retriever for standalone zero-dependency execution.
    Computes cosine similarity over normalized bag-of-words/n-gram vectors.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        # Add bigrams for phrase matching
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    @classmethod
    def get_vector(cls, text: str) -> Dict[str, float]:
        tokens = cls._tokenize(text)
        counts = Counter(tokens)
        norm = math.sqrt(sum(c * c for c in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    @classmethod
    def cosine_similarity(cls, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        return sum(vec1[k] * vec2[k] for k in intersection)


# ==============================================================================
# MemoryBank Engine
# ==============================================================================

class MemoryBank:
    """
    Unified long-term memory engine implementing:
    - Layered storage (conversations, event summaries, user portrait)
    - Ebbinghaus decay and reinforcement
    - Semantic retrieval
    """

    def __init__(
        self,
        forgetting_threshold: float = 0.15,
        user_name: str = "User",
    ):
        self.forgetting_threshold = forgetting_threshold
        self.memories: List[MemoryPiece] = []
        self.user_portrait = UserPortrait(name=user_name)
        self.event_summary = EventSummary()
        self._counter = 0

    def add_memory(
        self,
        content: str,
        current_day: float,
        category: str = "dialogue",
        initial_strength: float = 1.0,
    ) -> MemoryPiece:
        """Add a new memory piece into the storage."""
        self._counter += 1
        m_id = f"mem_{self._counter:03d}"
        piece = MemoryPiece(
            id=m_id,
            content=content,
            created_day=current_day,
            last_accessed_day=current_day,
            strength=initial_strength,
            category=category,
        )
        self.memories.append(piece)
        return piece

    def record_turn(self, current_day: float, user_input: str, ai_response: str) -> MemoryPiece:
        """Record a conversational turn as a memory piece."""
        turn_text = f"User: {user_input} -> AI: {ai_response}"
        return self.add_memory(turn_text, current_day=current_day, category="dialogue")

    def retrieve(
        self,
        query: str,
        current_day: float,
        top_k: int = 3,
        apply_forgetting: bool = True,
    ) -> List[Tuple[MemoryPiece, float, float]]:
        """
        Retrieve relevant memories for a query.
        Returns: List of tuples (memory_piece, semantic_score, retention_R).
        
        If apply_forgetting is True, memories whose retention R has dropped below
        forgetting_threshold are considered forgotten (or filtered out).
        """
        query_vec = SimpleVectorRetriever.get_vector(query)
        candidates: List[Tuple[MemoryPiece, float, float]] = []

        for m in self.memories:
            r = m.get_retention(current_day)
            # Check forgetting threshold
            if apply_forgetting and r < self.forgetting_threshold:
                continue

            mem_vec = SimpleVectorRetriever.get_vector(m.content)
            sim = SimpleVectorRetriever.cosine_similarity(query_vec, mem_vec)

            # Combined score = semantic similarity weighted by retention
            effective_score = sim * r if apply_forgetting else sim
            if sim > 0.05:
                candidates.append((m, sim, r))

        # Sort by effective score
        candidates.sort(key=lambda x: (x[1] * x[2]), reverse=True)
        top_results = candidates[:top_k]

        # Reinforce retrieved memories (Spacing Effect)
        for mem, _, _ in top_results:
            mem.reinforce(current_day)

        return top_results

    def format_prompt(
        self,
        current_query: str,
        current_day: float,
        top_k: int = 2,
    ) -> str:
        """
        Assemble the final MemoryBank-augmented prompt for the LLM:
        [Global User Portrait] + [Global Event Summary] + [Retrieved Memories] + [Query]
        """
        retrieved = self.retrieve(current_query, current_day=current_day, top_k=top_k)

        lines = [
            "============================================================",
            "### SYSTEM: SILICONFRIEND CONVERSATION PROMPT (WITH MEMORYBANK)",
            "============================================================",
            "",
            "--- [GLOBAL USER PORTRAIT] ---",
            self.user_portrait.format_portrait(),
            "",
            "--- [HIERARCHICAL EVENT SUMMARY] ---",
            self.event_summary.global_summary or "(No prior events summarized yet)",
            "",
            f"--- [RETRIEVED MEMORIES (Day {current_day})] ---",
        ]

        if not retrieved:
            lines.append("(No relevant memories recalled from MemoryBank)")
        else:
            for mem, sim, ret in retrieved:
                lines.append(
                    f"• [{mem.id}] (Created Day {mem.created_day}, Retention R={ret:.2f}, Strength S={mem.strength:.1f}):"
                )
                lines.append(f"  {mem.content}")

        lines.extend([
            "",
            "--- [CURRENT QUERY] ---",
            f"User: {current_query}",
            "AI Companion:",
        ])

        return "\n".join(lines)
