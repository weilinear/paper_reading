#!/usr/bin/env python3
"""
Generate Google Colab compatible Jupyter notebooks for paper_reading demos.
"""

import json
from pathlib import Path

notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
notebooks_dir.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "NONE",
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


# ==============================================================================
# Notebook 1: Poppler Scientific Paper Reader
# ==============================================================================
nb1_cells = [
    md_cell(
        """# Layout-Aware Scientific Paper PDF Reader
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/01_paper_reader_demo.ipynb)

This demo illustrates layout-aware multi-column PDF reading using Poppler (`pdftotext`, `pdfinfo`, `pdftoppm`), ensuring two-column scientific papers are read in 100% correct human reading order without column interleaving."""
    ),
    code_cell(
        """# 1. Install Poppler & Setup Repo
!apt-get update -qq && apt-get install -y -qq poppler-utils
![ -d paper_reading ] || git clone https://github.com/weilinear/paper_reading.git
%cd paper_reading"""
    ),
    code_cell(
        """# 2. Inspect a 2-Column Scientific Paper (ResNet CVPR 2016)
!curl -sL https://arxiv.org/pdf/1512.03385.pdf -o test_resnet.pdf
from pdf_reader import PDFReader

reader = PDFReader("test_resnet.pdf")
print("Document Metadata & Layout:")
for k, v in reader.get_info().items():
    print(f"  {k}: {v}")"""
    ),
    code_cell(
        """# 3. Extract Document Outline (TOC)
outline = reader.get_outline()
print("Document Outline:")
for sec in outline:
    indent = "  " * (sec.level - 1)
    print(f"{indent}- {sec.title} (Page {sec.page_num})")"""
    ),
    code_cell(
        """# 4. Read Two-Column Pages with Perfect Reading Order
# Left column body text flows continuously BEFORE right column body text!
page2_md = reader.read(pages=[2])
print(page2_md[:1500])"""
    ),
    code_cell(
        """# 5. Extract a Specific Section (e.g. Related Work)
print(reader.get_section("Related Work"))"""
    ),
    code_cell(
        """# 6. Render High-Resolution PNG for Visual Inspection
from IPython.display import Image, display

img_path = reader.render_page(page_num=2, dpi=150, output_path="resnet_page2.png")
display(Image("resnet_page2.png", width=650))"""
    ),
]

with open(notebooks_dir / "01_paper_reader_demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)


# ==============================================================================
# Notebook 2: Procedural Graphs & HotpotQA Benchmark
# ==============================================================================
nb2_cells = [
    md_cell(
        """# Procedural Graphs: Multi-Hop Reasoning on HotpotQA
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/02_hotpotqa_procedural_graph.ipynb)

Reproduction of HotpotQA multi-hop reasoning from **'Procedural Graphs: Self-Evolving Execution Structures for LLM Agents'** (Lu et al., 2026).
Compares:
1. **Vanilla ReAct** (Unguided baseline)
2. **Procedural Graph (Mode 1: Hand-crafted Expert Prior)**
3. **Procedural Graph (Mode 5: Scratch + Online Evolution)**"""
    ),
    code_cell(
        """# 1. Setup Environment
![ -d paper_reading ] || git clone https://github.com/weilinear/paper_reading.git
%cd paper_reading"""
    ),
    code_cell(
        """# 2. Inspect Procedural Graph Representation
from agents.procedural_graph import get_hotpotqa_expert_graph

expert_g = get_hotpotqa_expert_graph()
print(f"Expert Graph contains {len(expert_g.nodes)} nodes and {len(expert_g.edges)} edges.")

# Inspect localized 2-hop context around Step 1
print(expert_g.format_serialized_context("First_Hop_Retrieve", h=2))"""
    ),
    code_cell(
        """# 3. Run Benchmark Evaluation on Official HotpotQA Samples
from benchmarks.hotpotqa import HotpotQABenchmark, HotpotQAEnv
from evaluations.run_hotpotqa import run_evaluation, build_simulated_llm

bm = HotpotQABenchmark()
env = HotpotQAEnv()
llm = build_simulated_llm()

# Run evaluation across all 3 modes
results = run_evaluation(bm, env, llm, mode="all", split="dev", limit=20)"""
    ),
    code_cell(
        """# 4. Compare Sample Trajectories
# Observe how Vanilla ReAct cuts corners on Hop 1, whereas Procedural Graph enforces the 2nd hop
from agents.react import ReActAgent
from agents.procedural_graph import OnlineGuidanceEngine

task = bm.load_data(split="dev", limit=1)[0]
solver = ReActAgent(llm=llm)

print("--- [Task Question] ---")
print(task.question)
print("Gold Answer:", task.gold_answer)

# Run Vanilla
env.reset()
traj_v = solver.solve(task, env)
print("\\n=== Vanilla ReAct Trajectory (Unguided) ===")
print(traj_v.format_trajectory())
print("Final Prediction:", traj_v.final_answer)

# Run with Procedural Graph
engine = OnlineGuidanceEngine(graph=expert_g, h_hops=2)
provider = lambda traj, act: engine.get_guidance(traj.question, traj.format_trajectory(), act, step_count=len(traj.steps))
env.reset()
traj_pg = solver.solve(task, env, guidance_provider=provider)
print("\\n=== Procedural Graph Trajectory (Guided) ===")
print(traj_pg.format_trajectory())
print("Final Prediction:", traj_pg.final_answer)"""
    ),
]

with open(notebooks_dir / "02_hotpotqa_procedural_graph.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)


# ==============================================================================
# Notebook 3: MemoryBank & SiliconFriend
# ==============================================================================
nb3_cells = [
    md_cell(
        """# MemoryBank: Long-Term Memory for LLMs with Ebbinghaus Forgetting Curve
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/03_memory_bank_demo.ipynb)

Interactive demonstration of **MemoryBank** (Zhong et al., AAAI 2024 / arXiv:2305.10250):
- 3-Tiered Storage (Dialogue log, Hierarchical Event Summary, User Personality Portrait).
- Ebbinghaus Forgetting Curve ($R = e^{-t / S}$) & Spacing Effect Reinforcement ($S \\leftarrow S + 1, t \\leftarrow 0$).
- Dynamic Prompt Assembly for the SiliconFriend AI Companion."""
    ),
    code_cell(
        """# 1. Setup Environment
![ -d paper_reading ] || git clone https://github.com/weilinear/paper_reading.git
%cd paper_reading"""
    ),
    code_cell(
        """# 2. Visualize Ebbinghaus Forgetting Curves with Matplotlib
import numpy as np
import matplotlib.pyplot as plt

days = np.linspace(0, 10, 100)
plt.figure(figsize=(9, 5))

# Plot forgetting curves for different memory strengths
for S in [1.0, 2.0, 3.0, 4.0]:
    R = np.exp(-days / S)
    label = f"S = {S:.0f} ({'Initial memory' if S == 1 else f'Recalled {int(S-1)} time(s)'})"
    plt.plot(days, R, label=label, linewidth=2)

plt.axhline(y=0.15, color='red', linestyle='--', label='Forgetting Threshold (R = 0.15)')
plt.title("Ebbinghaus Forgetting Curve & Spacing Effect in MemoryBank", fontsize=13)
plt.xlabel("Elapsed Time t (Days since last recall)", fontsize=11)
plt.ylabel("Memory Retention R = exp(-t / S)", fontsize=11)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.show()"""
    ),
    code_cell(
        """# 3. Run the Multi-Day MemoryBank Interactive Simulation
from memory_bank import MemoryBank

# Initialize MemoryBank for Linda
bank = MemoryBank(forgetting_threshold=0.15, user_name="Linda")

# Day 1: Add initial conversations
bank.record_turn(1.0, "I want to learn Python. What book do you recommend?", "I suggest 'Automate the Boring Stuff with Python'.")
bank.record_turn(1.0, "I had a ham sandwich for lunch today.", "Sounds tasty! Enjoy your break.")
bank.event_summary.add_daily_summary(1, "Linda asked for Python learning resources (book: Automate the Boring Stuff).")

# Day 2: Emotional interaction & User portrait update
bank.record_turn(2.0, "I feel stressed at work and anxious about learning.", "Take it step by step; you're doing great.")
bank.user_portrait.traits = ["introverted", "ambitious", "growth-oriented"]
bank.user_portrait.interests = ["Python programming", "reading"]
bank.user_portrait.emotional_state = "experiencing workplace learning anxiety"
bank.event_summary.add_daily_summary(2, "Linda expressed workplace learning stress and received encouragement.")

print("Memory Status on Day 2.0:")
for m in bank.memories:
    print(f"• [{m.id}] S={m.strength:.1f}, R={m.get_retention(2.0):.2f}: {m.content[:45]}...")"""
    ),
    code_cell(
        """# 4. Day 4: Query triggers the Spacing Effect!
# Query recalling the book on Day 4.0
query = "What was the Python book you recommended?"
retrieved = bank.retrieve(query, current_day=4.0, top_k=1, apply_forgetting=False)

recalled_mem = retrieved[0][0]
print(f"✓ Recalled: [{recalled_mem.id}] -> New Strength S = {recalled_mem.strength:.1f} (Reinforced!)")"""
    ),
    code_cell(
        """# 5. Day 7: Divergence between Reinforced vs Unreinforced Memories
print("Memory Retention Comparison on Day 7.0:")
for m in bank.memories:
    r = m.get_retention(7.0)
    status = "RETAINED" if r >= bank.forgetting_threshold else "FORGOTTEN"
    print(f"• [{m.id}] S={m.strength:.1f}, R={r:.4f} -> {status} ({m.content[:40]}...)")"""
    ),
    code_cell(
        """# 6. Final Prompt Assembled for the LLM Companion (SiliconFriend)
prompt = bank.format_prompt("Can you remind me of that Python book? How should I study it?", current_day=7.0)
print(prompt)"""
    ),
]

with open(notebooks_dir / "03_memory_bank_demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

print("Generated 3 Colab notebooks in notebooks/:")
for p in notebooks_dir.glob("*.ipynb"):
    print("  -", p.name)
