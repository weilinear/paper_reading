"""
Unified LLM interface for agent execution and testing.
Supports OpenAI-compatible APIs (DeepSeek, OpenAI, Gemini, Local models)
as well as a MockLLM for offline deterministic testing.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response given a text prompt."""
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible client using urllib (zero extra pip dependencies).
    Works with OpenAI, DeepSeek, Gemini (OpenAI compat), Ollama, vLLM, etc.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


class MockLLM(BaseLLM):
    """
    Deterministic rule-based / scripted LLM for offline tests and verification.
    """

    def __init__(self, responder: Optional[Callable[[str], str]] = None):
        self.responder = responder
        self.calls: List[str] = []

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.calls.append(prompt)
        if self.responder:
            res = self.responder(prompt)
        else:
            res = "Thought: I should search for information.\nAction: Search[None]"

        if stop:
            for s in stop:
                if s in res:
                    res = res.split(s)[0]
        return res
