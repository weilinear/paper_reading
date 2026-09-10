"""
Standard Procedural Graph configurations from the paper:
- Mode 1: Hand-crafted Expert Prior (HotpotQA)
- Mode 4/5: Minimal Skeleton (Start -> End)
"""

from agents.procedural_graph.graph import ProceduralGraph


def get_hotpotqa_expert_graph() -> ProceduralGraph:
    """
    Construct the HotpotQA Mode 1 Expert Prior graph described in Section 5.3 and Appendix B.4.
    """
    g = ProceduralGraph(name="hotpotqa_expert_prior")

    # Nodes
    g.add_node("Start", type="STATUS", description="Task initialized; awaiting first action.")
    g.add_node(
        "First_Hop_Retrieve",
        type="ACTION",
        description="Execute Search[entity] to fetch primary evidence passages.",
    )
    g.add_node(
        "Scan_Index",
        type="ACTION",
        description="Review retrieved passage via Lookup or inspection to locate candidate bridge terms.",
    )
    g.add_node(
        "Bridge_Extract",
        type="REASONING",
        description="Identify and isolate the connecting bridge entity or attribute.",
    )
    g.add_node(
        "Second_Hop_Retrieve",
        type="ACTION",
        description="Execute Search[bridge_entity] to retrieve the second supporting document.",
    )
    g.add_node(
        "Compare_Answer",
        type="REASONING",
        description="Cross-reference facts from both documents to verify and deduce the final answer.",
    )
    g.add_node(
        "Finish",
        type="ACTION",
        description="Submit the verified answer via Finish[answer].",
    )
    g.add_node("End", type="STATUS", description="Task completed.")

    # Edges
    g.add_edge(
        source="Start",
        target="First_Hop_Retrieve",
        relation="LEADS_TO",
        condition="New question received",
        guidance="Identify the core named entity in the question and call Search[entity].",
        pitfalls="Do not search the entire multi-hop question as a sentence; search specific named entities.",
    )
    g.add_edge(
        source="First_Hop_Retrieve",
        target="Scan_Index",
        relation="LEADS_TO",
        condition="First passage retrieved",
        guidance="Review the retrieved primary passage to locate specific bridge terms (birth dates, locations, directors, actors).",
        pitfalls="Do not skip reading evidence details; missing the exact bridge entity name causes second-hop search failure.",
    )
    g.add_edge(
        source="Scan_Index",
        target="Bridge_Extract",
        relation="PROVIDES_INPUT_FOR",
        condition="Passage contains candidate entity",
        guidance="Extract the explicit connecting entity or bridge term linking the first passage to the target question.",
        pitfalls="Ensure the extracted bridge term matches exact Wikipedia capitalization conventions.",
    )
    g.add_edge(
        source="Bridge_Extract",
        target="Second_Hop_Retrieve",
        relation="LEADS_TO",
        condition="Bridge entity identified",
        guidance="Call Search[bridge_entity] to retrieve facts needed to resolve the remaining constraint.",
        pitfalls="Do not guess the answer without verifying second-hop source documentation.",
    )
    g.add_edge(
        source="Second_Hop_Retrieve",
        target="Compare_Answer",
        relation="PROVIDES_INPUT_FOR",
        condition="Both hops retrieved",
        guidance="Cross-reference dates, titles, nationalities, or positions between Hop 1 and Hop 2.",
        pitfalls="Do not confuse attributes of the bridge entity with attributes of the main subject.",
    )
    g.add_edge(
        source="Compare_Answer",
        target="Finish",
        relation="LEADS_TO",
        condition="Answer deduced",
        guidance="Formulate a concise, exact-match answer string and invoke Finish[answer].",
        pitfalls="Do not output conversational boilerplate or full sentences inside Finish[...]; output only the concise entity or value.",
    )
    g.add_edge(
        source="Finish",
        target="End",
        relation="TRANSITIONS_TO",
        condition="Answer submitted",
        guidance="Task execution finished successfully.",
        pitfalls="",
    )

    return g


def get_skeleton_graph() -> ProceduralGraph:
    """
    Construct the minimal skeleton graph (Start -> End) used for evolution from scratch (Mode 4/5).
    """
    g = ProceduralGraph(name="minimal_skeleton")
    g.add_node("Start", type="STATUS", description="Task initialized.")
    g.add_node("End", type="STATUS", description="Task finished.")
    g.add_edge(
        source="Start",
        target="End",
        relation="LEADS_TO",
        condition="Task execution",
        guidance="Complete task by selecting appropriate tools.",
        pitfalls="Avoid unproductive loops.",
    )
    return g
