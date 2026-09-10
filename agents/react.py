"""
ReAct (Reasoning + Acting) Agent Solver.
Supports unconstrained generation (Vanilla ReAct) as well as
dynamic situational guidance injection (Procedural Graph).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.llm import BaseLLM
from benchmarks.base import Task
from benchmarks.hotpotqa.environment import HotpotQAEnv


@dataclass
class StepRecord:
    step: int
    thought: str
    action: str
    observation: str
    guidance: Optional[str] = None


@dataclass
class Trajectory:
    task_id: str
    question: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    reward: float = 0.0

    def format_trajectory(self) -> str:
        """Format trajectory as string."""
        lines = []
        for s in self.steps:
            lines.append(f"Thought {s.step}: {s.thought}")
            lines.append(f"Action {s.step}: {s.action}")
            lines.append(f"Observation {s.step}: {s.observation}")
        return "\n".join(lines)


REACT_INSTRUCTION = """Solve a multi-step question answering task by interleaving Thought, Action, and Observation steps.
Thought can reason about the current situation.
Action can be one of three types:
(1) Search[entity], which searches Wikipedia and returns the first few sentences.
(2) Lookup[keyword], which returns the next sentence containing the keyword in the currently opened page.
(3) Finish[answer], which returns the final answer and finishes the task.

Example:
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny and find the area that the eastern sector extends into, then get its elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: [Colorado orogeny]: The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.
Thought 2: The eastern sector extends into the High Plains. I should search High Plains.
Action 2: Search[High Plains]
Observation 2: [High Plains]: High Plains refers to one of two distinct land regions: a subregion of the Great Plains in the US, ranging in elevation from 1,800 to 7,000 ft.
Thought 3: The elevation range is 1,800 to 7,000 ft.
Action 3: Finish[1,800 to 7,000 ft]
"""


class ReActAgent:
    """Standard ReAct agent loop with optional procedural guidance injection."""

    def __init__(self, llm: BaseLLM, max_steps: int = 7):
        self.llm = llm
        self.max_steps = max_steps

    def solve(
        self,
        task: Task,
        env: HotpotQAEnv,
        guidance_provider: Optional[Callable[[Trajectory, str], str]] = None,
    ) -> Trajectory:
        """Execute the agent on a task."""
        env.reset(task.context)
        trajectory = Trajectory(task_id=task.id, question=task.question)

        prompt_history = f"{REACT_INSTRUCTION}\nQuestion: {task.question}\n"

        for step_idx in range(1, self.max_steps + 1):
            # Optional procedural guidance injection
            guidance_text = None
            if guidance_provider is not None:
                recent_action = trajectory.steps[-1].action if trajectory.steps else "Start"
                guidance_text = guidance_provider(trajectory, recent_action)

            # Build prompt
            current_prompt = prompt_history
            if guidance_text:
                current_prompt += f"\n[Situational Procedural Guidance]:\n{guidance_text}\n"

            current_prompt += f"Thought {step_idx}:"

            response = self.llm.generate(current_prompt, stop=[f"\nObservation {step_idx}:", "\nObservation:"])
            response = response.strip()

            # Parse Thought and Action
            thought, action = self._parse_thought_and_action(response, step_idx)

            # Execute action in environment
            obs, is_finished = env.step(action)

            # Record step
            step_record = StepRecord(
                step=step_idx,
                thought=thought,
                action=action,
                observation=obs,
                guidance=guidance_text,
            )
            trajectory.steps.append(step_record)

            prompt_history += f"Thought {step_idx}: {thought}\nAction {step_idx}: {action}\nObservation {step_idx}: {obs}\n"

            if is_finished or env.is_finished:
                trajectory.final_answer = env.final_answer or self._extract_finish_arg(action)
                break

        return trajectory

    @staticmethod
    def _parse_thought_and_action(response: str, step_idx: int) -> tuple[str, str]:
        """Extract thought and action from LLM generation."""
        # Clean response
        response = response.replace(f"Thought {step_idx}:", "").strip()

        action_m = re.search(r"Action(?:\s+\d+)?:?\s*(.+)$", response, re.MULTILINE | re.DOTALL)
        if action_m:
            thought = response[: action_m.start()].strip()
            action_raw = action_m.group(1).strip().split("\n")[0].strip()
        else:
            # Fallback if no Action label
            thought = response
            action_raw = "Finish[unknown]"

        return thought, action_raw

    @staticmethod
    def _extract_finish_arg(action_str: str) -> str:
        m = re.match(r"^Finish\[(.*)\]$", action_str.strip(), re.DOTALL)
        return m.group(1).strip() if m else action_str
