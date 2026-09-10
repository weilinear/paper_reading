"""
Online Generative Guidance Engine.
Implements locate, extract, and generate operations (Section 3.2).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents.llm import BaseLLM
from agents.procedural_graph.graph import ProceduralGraph


class OnlineGuidanceEngine:
    """
    Generates step-level situational guidance by:
    1. Localizing the agent's current active node u_t in the graph.
    2. Extracting the h-hop directed topological neighborhood G_t.
    3. Translating the subgraph attributes into situational guidance g_t.
    """

    def __init__(
        self,
        graph: ProceduralGraph,
        h_hops: int = 2,
        guidance_llm: Optional[BaseLLM] = None,
    ):
        self.graph = graph
        self.h_hops = h_hops
        self.guidance_llm = guidance_llm

    def localize_node(self, recent_action: str, step_count: int = 0) -> str:
        """
        Map agent's recent action/state to active node in graph (Match operation).
        """
        action_clean = recent_action.strip().lower()

        if action_clean in ("start", "none", "") or step_count == 0:
            return "Start"

        if action_clean.startswith("finish"):
            return "Finish"

        if action_clean.startswith("lookup"):
            return "Scan_Index"

        if action_clean.startswith("search"):
            # If this is the second or subsequent search, localize to Second_Hop_Retrieve
            if step_count >= 2:
                return "Second_Hop_Retrieve"
            return "First_Hop_Retrieve"

        # Direct node ID match
        for node_id in self.graph.nodes.keys():
            if node_id.lower() in action_clean:
                return node_id

        # Fallback to Start
        return "Start"

    def get_guidance(self, query: str, recent_context: str, recent_action: str, step_count: int = 0) -> str:
        """
        Generate situational guidance for the current decision step.
        """
        active_node = self.localize_node(recent_action, step_count=step_count)
        serialized_context = self.graph.format_serialized_context(active_node, h=self.h_hops)

        # If a guidance LLM is configured, run the translation prompt
        if self.guidance_llm is not None:
            prompt = f"""You are an expert cognitive architect and execution guide for an AI agent solving the task: Multi-hop Question Answering.

Here is the local Procedural Graph context:
{serialized_context}

Here is the current user question / active query:
{query}

Here is the agent's recent execution trajectory:
{recent_context}

Analyze this graph context in the context of the agent's current progress. Using the condition, guidance, and pitfalls attributes, generate clear, concise, and actionable guidance advising the agent on what step to pursue next and what pitfalls to avoid. Keep guidance under 3 sentences."""
            return self.guidance_llm.generate(prompt).strip()

        # Direct structured guidance from graph attributes (deterministic & fast)
        outgoing = self.graph.get_outgoing_edges(active_node)
        if not outgoing:
            # Check if any transitions from Start or fallback
            outgoing = self.graph.edges[:1]

        guidance_parts = []
        for e in outgoing:
            target_desc = self.graph.nodes.get(e.target, {}).description if e.target in self.graph.nodes else ""
            part = f"• Next Admissible Step: [{e.target}]"
            if target_desc:
                part += f" ({target_desc})"
            if e.guidance:
                part += f"\n  Guidance: {e.guidance}"
            if e.pitfalls:
                part += f"\n  Pitfall to avoid: {e.pitfalls}"
            guidance_parts.append(part)

        return "\n".join(guidance_parts)
