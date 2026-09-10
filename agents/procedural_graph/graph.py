"""
Procedural Graph Data Structure.
Formalizes procedural knowledge into directed, attributed triplets:
(procedure, relation, procedure) with (condition, guidance, pitfalls).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Node:
    id: str
    type: str = "ACTION"  # ACTION, STATUS, REASONING
    description: str = ""


@dataclass
class Edge:
    source: str
    target: str
    relation: str = "LEADS_TO"  # LEADS_TO, PROVIDES_INPUT_FOR, REQUIRES, TRANSITIONS_TO
    condition: Optional[str] = None
    guidance: str = ""
    pitfalls: str = ""


class ProceduralGraph:
    """Directed multigraph of procedural knowledge with rich edge attributes."""

    def __init__(self, name: str = "procedural_graph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def add_node(self, id: str, type: str = "ACTION", description: str = "") -> Node:
        node = Node(id=id, type=type, description=description)
        self.nodes[id] = node
        return node

    def delete_node(self, id: str) -> bool:
        if id in self.nodes:
            del self.nodes[id]
            self.edges = [e for e in self.edges if e.source != id and e.target != id]
            return True
        return False

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "LEADS_TO",
        condition: Optional[str] = None,
        guidance: str = "",
        pitfalls: str = "",
    ) -> Edge:
        # Auto-create nodes if missing
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)

        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            condition=condition,
            guidance=guidance,
            pitfalls=pitfalls,
        )
        self.edges.append(edge)
        return edge

    def delete_edges(self, source: str, target: str) -> int:
        initial_len = len(self.edges)
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        return initial_len - len(self.edges)

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def get_h_hop_neighborhood(self, node_id: str, h: int = 2) -> Dict[str, Any]:
        """
        Extract directed h-hop neighborhood starting from node_id.
        Returns active node info, hop-1 transitions, and hop-2 horizon.
        """
        if node_id not in self.nodes:
            # Fallback to full graph
            return self.to_dict()

        visited_nodes: Set[str] = {node_id}
        current_frontier = {node_id}
        edges_by_hop: Dict[int, List[Edge]] = {}

        for hop in range(1, h + 1):
            next_frontier = set()
            edges_by_hop[hop] = []
            for curr in current_frontier:
                for edge in self.get_outgoing_edges(curr):
                    edges_by_hop[hop].append(edge)
                    next_frontier.add(edge.target)
            visited_nodes.update(next_frontier)
            current_frontier = next_frontier
            if not current_frontier:
                break

        return {
            "active_node": asdict(self.nodes[node_id]),
            "hop_1_edges": [asdict(e) for e in edges_by_hop.get(1, [])],
            "hop_2_edges": [asdict(e) for e in edges_by_hop.get(2, [])],
            "all_visited_nodes": [asdict(self.nodes[n]) for n in visited_nodes if n in self.nodes],
        }

    def format_serialized_context(self, node_id: str, h: int = 2) -> str:
        """
        Format the local h-hop graph context into standard serialized text,
        matching the exact paper representation (Section 3.2, Appendix B.4).
        """
        data = self.get_h_hop_neighborhood(node_id, h=h)
        active = data.get("active_node") or (asdict(self.nodes[node_id]) if node_id in self.nodes else None)

        if not active:
            return "No procedural graph context available."

        lines = [
            f"Active Cognitive Node: [{active['id']}] (Type: {active['type']})",
            f"Description: {active['description']}",
            "",
            "Immediate Transition Options (Hop 1):",
        ]

        hop1 = data.get("hop_1_edges", [])
        if not hop1:
            lines.append("  (No direct outgoing transitions)")
        for e in hop1:
            cond_str = f"Condition: {e['condition']}" if e['condition'] else "Unconditional"
            lines.append(f"- Transition: [{e['source']}] → [{e['target']}] ({cond_str})")
            if e["guidance"]:
                lines.append(f"  * Guidance: {e['guidance']}")
            if e["pitfalls"]:
                lines.append(f"  * Pitfalls to Avoid: {e['pitfalls']}")

        hop2 = data.get("hop_2_edges", [])
        if hop2:
            lines.append("\nSubsequent Horizon (Hop 2):")
            for e in hop2:
                cond_str = f"Condition: {e['condition']}" if e['condition'] else "Unconditional"
                lines.append(f"- Transition: [{e['source']}] → [{e['target']}] ({cond_str})")
                if e["guidance"]:
                    lines.append(f"  * Guidance: {e['guidance']}")
                if e["pitfalls"]:
                    lines.append(f"  * Pitfalls to Avoid: {e['pitfalls']}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProceduralGraph:
        graph = cls(name=data.get("name", "procedural_graph"))
        for n in data.get("nodes", []):
            graph.add_node(id=n["id"], type=n.get("type", "ACTION"), description=n.get("description", ""))
        for e in data.get("edges", []):
            graph.add_edge(
                source=e["source"],
                target=e["target"],
                relation=e.get("relation", "LEADS_TO"),
                condition=e.get("condition"),
                guidance=e.get("guidance", ""),
                pitfalls=e.get("pitfalls", ""),
            )
        return graph

    @classmethod
    def from_json(cls, json_str: str) -> ProceduralGraph:
        return cls.from_dict(json.loads(json_str))
