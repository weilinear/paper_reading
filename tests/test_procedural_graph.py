#!/usr/bin/env python3
"""
Unit tests for Procedural Graph representation, online guidance, and offline self-evolution.
"""

import unittest

from agents.procedural_graph import (
    MutationEdits,
    OfflineRefiner,
    OnlineGuidanceEngine,
    ProceduralGraph,
    get_hotpotqa_expert_graph,
    get_skeleton_graph,
)


class TestProceduralGraph(unittest.TestCase):
    def test_graph_construction(self):
        g = ProceduralGraph(name="test_graph")
        g.add_node("A", type="ACTION", description="Step A")
        g.add_node("B", type="ACTION", description="Step B")
        g.add_edge(
            source="A",
            target="B",
            relation="LEADS_TO",
            condition="When ready",
            guidance="Proceed from A to B",
            pitfalls="Do not skip A",
        )

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 1)

        # Outgoing edges
        outgoing = g.get_outgoing_edges("A")
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(outgoing[0].target, "B")

        # Serialized context
        context_str = g.format_serialized_context("A", h=1)
        self.assertIn("Active Cognitive Node: [A]", context_str)
        self.assertIn("Guidance: Proceed from A to B", context_str)
        self.assertIn("Pitfalls to Avoid: Do not skip A", context_str)

    def test_hotpotqa_expert_graph(self):
        expert_g = get_hotpotqa_expert_graph()
        self.assertIn("Start", expert_g.nodes)
        self.assertIn("First_Hop_Retrieve", expert_g.nodes)
        self.assertIn("Second_Hop_Retrieve", expert_g.nodes)
        self.assertIn("Finish", expert_g.nodes)

        # 2-hop neighborhood of Start
        sub = expert_g.get_h_hop_neighborhood("Start", h=2)
        self.assertEqual(sub["active_node"]["id"], "Start")
        self.assertTrue(len(sub["hop_1_edges"]) >= 1)

    def test_online_guidance_engine(self):
        expert_g = get_hotpotqa_expert_graph()
        engine = OnlineGuidanceEngine(graph=expert_g, h_hops=2)

        # Step 1: Start
        node_1 = engine.localize_node("Start", step_count=0)
        self.assertEqual(node_1, "Start")
        g1 = engine.get_guidance("What position was held?", "", "Start", step_count=0)
        self.assertIn("First_Hop_Retrieve", g1)

        # Step 2: After first search
        node_2 = engine.localize_node("Search[Kiss and Tell]", step_count=1)
        self.assertEqual(node_2, "First_Hop_Retrieve")

        # Step 3: After second search
        node_3 = engine.localize_node("Search[Shirley Temple]", step_count=2)
        self.assertEqual(node_3, "Second_Hop_Retrieve")

    def test_offline_refiner_and_validation_gate(self):
        skeleton = get_skeleton_graph()
        self.assertEqual(len(skeleton.nodes), 2)

        refiner = OfflineRefiner()
        edits = MutationEdits(
            add_nodes=[{"id": "Verify_Step", "type": "ACTION", "description": "Verification"}],
            add_edges=[{"source": "Start", "target": "Verify_Step", "guidance": "Verify first"}],
        )

        mutated = refiner.apply_edits(skeleton, edits)
        self.assertIn("Verify_Step", mutated.nodes)
        self.assertEqual(len(mutated.edges), 2)

        # Validation gate: Candidate improves score (commit)
        committed, result_graph, cand_s, curr_s = refiner.validation_gate(
            candidate_graph=mutated,
            current_graph=skeleton,
            eval_fn=lambda g: 80.0 if "Verify_Step" in g.nodes else 50.0,
            edits=edits,
        )
        self.assertTrue(committed)
        self.assertIn("Verify_Step", result_graph.nodes)

        # Validation gate: Candidate degrades score (reject & log to rejection memory)
        committed, result_graph, cand_s, curr_s = refiner.validation_gate(
            candidate_graph=mutated,
            current_graph=skeleton,
            eval_fn=lambda g: 30.0 if "Verify_Step" in g.nodes else 50.0,
            edits=edits,
        )
        self.assertFalse(committed)
        self.assertNotIn("Verify_Step", result_graph.nodes)
        self.assertEqual(len(refiner.rejection_memory), 1)


if __name__ == "__main__":
    unittest.main()
