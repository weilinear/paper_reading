"""
HotpotQA Execution Environment.
Provides the classic ReAct action space:
- Search[entity]: Retrieve Wikipedia page summary for the entity.
- Lookup[keyword]: Scan the current active document for the keyword.
- Finish[answer]: Conclude execution with the final answer.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple


class HotpotQAEnv:
    """
    Simulation environment for HotpotQA multi-hop reasoning.
    Supports local passage dictionary (gold and distractor contexts)
    for fast deterministic evaluations without external network latency.
    """

    def __init__(self, passages: Optional[Dict[str, str]] = None):
        """
        passages: Dict mapping article titles to full text (or list of sentences).
        """
        self.passages: Dict[str, str] = passages or {}
        # Lowercase mapping for robust case-insensitive matching
        self._lower_map = {k.lower(): k for k in self.passages.keys()}

        self.current_title: Optional[str] = None
        self.current_text: Optional[str] = None
        self.current_sentences: List[str] = []
        self.lookup_idx: int = 0
        self.is_finished: bool = False
        self.final_answer: Optional[str] = None
        self.steps_taken: int = 0

    def reset(self, passages: Optional[Dict[str, str]] = None) -> None:
        """Reset environment state for a new question."""
        if passages is not None:
            self.passages = passages
            self._lower_map = {k.lower(): k for k in self.passages.keys()}
        self.current_title = None
        self.current_text = None
        self.current_sentences = []
        self.lookup_idx = 0
        self.is_finished = False
        self.final_answer = None
        self.steps_taken = 0

    def step(self, action_str: str) -> Tuple[str, bool]:
        """
        Parse and execute an action string, e.g.:
        - 'Search[entity]'
        - 'Lookup[keyword]'
        - 'Finish[answer]'
        Returns: (observation: str, is_finished: bool)
        """
        self.steps_taken += 1
        action_str = action_str.strip()

        # Parse Action[Argument]
        m = re.match(r"^(\w+)\s*\[(.*)\]$", action_str, re.DOTALL)
        if not m:
            # Fallback for formats like Action: Argument or search(argument)
            m2 = re.match(r"^(\w+)\s*[:\(]\s*[\"']?(.*?)[\"']?\s*\)?$", action_str, re.DOTALL)
            if m2:
                cmd = m2.group(1).lower()
                arg = m2.group(2).strip()
            else:
                return f"Invalid action syntax: '{action_str}'. Expected Search[entity], Lookup[keyword], or Finish[answer].", False
        else:
            cmd = m.group(1).lower()
            arg = m.group(2).strip()

        if cmd == "search":
            obs = self._search(arg)
            return obs, False
        elif cmd == "lookup":
            obs = self._lookup(arg)
            return obs, False
        elif cmd == "finish":
            self.is_finished = True
            self.final_answer = arg
            return f"Answer submitted: {arg}", True
        else:
            return f"Unknown action: '{cmd}'. Available actions: Search[entity], Lookup[keyword], Finish[answer].", False

    def _search(self, entity: str) -> str:
        """Search Wikipedia entity."""
        entity_norm = entity.strip()
        entity_lower = entity_norm.lower()

        # Exact or case-insensitive match
        match_title = self._lower_map.get(entity_lower)

        if not match_title:
            # Substring match
            candidates = [k for k in self.passages.keys() if entity_lower in k.lower() or k.lower() in entity_lower]
            if not candidates:
                # Fuzzy match
                close = difflib.get_close_matches(entity_norm, self.passages.keys(), n=3, cutoff=0.5)
                if close:
                    candidates = close

            if candidates:
                match_title = candidates[0]
            else:
                return f"Could not find [{entity_norm}]. Similar titles: {list(self.passages.keys())[:3]}."

        self.current_title = match_title
        raw_text = self.passages[match_title]
        self.current_text = raw_text

        # Split into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
        self.current_sentences = sentences
        self.lookup_idx = 0

        # Return first 1-2 sentences as summary
        preview = " ".join(sentences[:2]) if sentences else raw_text[:200]
        return f"[{match_title}]: {preview}"

    def _lookup(self, keyword: str) -> str:
        """Scan current document for keyword."""
        if not self.current_sentences:
            return "No page currently open. Use Search[entity] first."

        keyword_lower = keyword.strip().lower()
        for idx in range(self.lookup_idx, len(self.current_sentences)):
            sent = self.current_sentences[idx]
            if keyword_lower in sent.lower():
                self.lookup_idx = idx + 1
                return f"[{self.current_title}] (result {idx + 1}/{len(self.current_sentences)}): {sent}"

        # If not found after lookup_idx, check from beginning
        if self.lookup_idx > 0:
            for idx in range(0, self.lookup_idx):
                sent = self.current_sentences[idx]
                if keyword_lower in sent.lower():
                    self.lookup_idx = idx + 1
                    return f"[{self.current_title}] (wrapped to result {idx + 1}/{len(self.current_sentences)}): {sent}"

        return f"No sentence containing '{keyword}' found in [{self.current_title}]."
