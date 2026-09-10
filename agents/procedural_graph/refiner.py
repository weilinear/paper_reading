"""
Offline Self-Evolution Refiner (Section 3.3).
Implements feedback-driven mutations, validation gating, and rejection memory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.procedural_graph.graph import Edge, Node, ProceduralGraph
from agents.react import Trajectory


@dataclass
class MutationEdits:
    add_nodes: List[Dict[str, Any]] = field(default_factory=list)
    delete_nodes: List[str] = field(default_factory=list)
    add_edges: List[Dict[str, Any]] = field(default_factory=list)
    delete_edges: List[Dict[str, Any]] = field(default_factory=list)


class OfflineRefiner:
    """
    Offline self-evolution engine that analyzes execution traces
    and iteratively mutates the Procedural Graph.
    """

    def __init__(self, validation_threshold: float = 0.0):
        self.rejection_memory: List[Dict[str, Any]] = []
        self.validation_threshold = validation_threshold

    def apply_edits(self, graph: ProceduralGraph, edits: MutationEdits) -> ProceduralGraph:
        """Apply structured topological edits to produce candidate graph."""
        candidate = copy.deepcopy(graph)

        # 1. Delete edges
        for de in edits.delete_edges:
            candidate.delete_edges(de["source"], de["target"])

        # 2. Delete nodes
        for dn in edits.delete_nodes:
            candidate.delete_node(dn)

        # 3. Add nodes
        for an in edits.add_nodes:
            candidate.add_node(
                id=an["id"],
                type=an.get("type", "ACTION"),
                description=an.get("description", ""),
            )

        # 4. Add edges (or update attributes)
        for ae in edits.add_edges:
            candidate.add_edge(
                source=ae["source"],
                target=ae["target"],
                relation=ae.get("relation", "LEADS_TO"),
                condition=ae.get("condition"),
                guidance=ae.get("guidance", ""),
                pitfalls=ae.get("pitfalls", ""),
            )

        return candidate

    def validation_gate(
        self,
        candidate_graph: ProceduralGraph,
        current_graph: ProceduralGraph,
        eval_fn: Callable[[ProceduralGraph], float],
        edits: Optional[MutationEdits] = None,
    ) -> Tuple[bool, ProceduralGraph, float, float]:
        """
        Step 3 & 4: Validation Gating and Rejection Memory.
        Commits candidate graph if val score >= current score.
        Otherwise logs candidate to rejection memory and rolls back.
        """
        curr_score = eval_fn(current_graph)
        cand_score = eval_fn(candidate_graph)

        if cand_score >= curr_score + self.validation_threshold:
            # Commit
            return True, candidate_graph, cand_score, curr_score
        else:
            # Reject and log to rejection memory
            self.rejection_memory.append({
                "rejected_graph": candidate_graph.to_dict(),
                "edits": edits.__dict__ if edits else {},
                "candidate_score": cand_score,
                "current_score": curr_score,
            })
            # Roll back to current graph
            return False, current_graph, cand_score, curr_score
