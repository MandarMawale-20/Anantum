"""Llama.cpp wrapper for local GGUF inference.

Handles model loading, prompt building, text generation (sync + streaming),
and JSON-structured planning for multi-step tasks.
"""

import time
import json
import re
from pathlib import Path
from typing import Generator


SYSTEM_PROMPT = """You are Anantum, a concise AI voice assistant running entirely offline on this device.

CRITICAL RULES:
1. Keep responses SHORT: 1-2 sentences for simple questions, max 4 for complex ones.
2. Never say "Certainly!", "Of course!", "Great question!", "Sure!" or any hollow opener.
3. Never hallucinate facts. If you don't know, say "I don't know" plainly.
4. Never suggest going online or checking the web. You are fully offline.
5. Never end with "Is there anything else I can help you with?" or similar.
6. If asked who you are: "I'm Anantum, a local AI assistant."
7. If asked about the user (name, location, preferences): use ONLY the memory facts below.
   Do NOT invent user details. If not in memory, say "I don't have that stored yet."
8. Respond naturally, like a human assistant would speak — not like a chatbot.

Current time: {current_time}
"""

MEMORY_BLOCK = """
What I know about this user (use ONLY these when answering personal questions):
{facts}
"""


def build_prompt(user_message: str, memory_context: dict,
                 conversation_history: list, mode: str = "normal") -> str:
    """Build a Gemma-style prompt with memory and recent history."""
    now = time.strftime("%I:%M %p, %A %B %d %Y")
    system = SYSTEM_PROMPT.format(current_time=now)

    facts = memory_context.get("relevant_facts", [])
    if facts:
        fact_lines = [f"- {f['text']}" for f in facts[:5]]
        if fact_lines:
            system += MEMORY_BLOCK.format(facts="\n".join(fact_lines))

    history = conversation_history[-8:]
    history_str = ""
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            history_str += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        else:
            history_str += f"<start_of_turn>model\n{content}<end_of_turn>\n"

    prompt = (
        f"<start_of_turn>user\n"
        f"{system}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"Understood. I'm Anantum, ready to help.\n"
        f"<end_of_turn>\n"
        f"{history_str}"
        f"<start_of_turn>user\n{user_message}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return prompt


class LLMManager:
    """Manages a local GGUF language model via llama.cpp."""

    def __init__(self, model_path: str, n_ctx: int = 2048,
                 n_threads: int = 4, n_gpu_layers: int = 0):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            from llama_cpp import Llama

            gpu_info = f"GPU ({self.n_gpu_layers} layers)" if self.n_gpu_layers > 0 else "CPU only"
            print(f"[LLM] Loading {self.model_path.name} on {gpu_info}...")
            t0 = time.time()

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=512,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                rope_scaling_type=1,
                rope_freq_scale=1.0,
            )
            elapsed = time.time() - t0
            self._loaded = True
            print(f"[LLM] Loaded in {elapsed:.1f}s  ({gpu_info})")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "For GPU (CUDA): CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall\n"
                "For CPU only:  pip install llama-cpp-python"
            )

    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.65) -> str:
        if not self._loaded:
            self.load()

        stop = ["<end_of_turn>", "<start_of_turn>", "\nUser:", "User:", "\nuser:"]

        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.2,
            stop=stop,
            echo=False,
        )
        text = out["choices"][0]["text"].strip()

        for stop_tok in stop:
            if stop_tok.strip() in text:
                text = text[:text.index(stop_tok.strip())].strip()

        return text

    def generate_streaming(self, prompt: str, max_tokens: int = 200,
                           temperature: float = 0.65) -> Generator[str, None, None]:
        if not self._loaded:
            self.load()

        stop = ["<end_of_turn>", "<start_of_turn>", "\nUser:", "User:", "\nuser:"]

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.2,
            stop=stop,
            echo=False,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                if any(s.strip() in token for s in stop):
                    return
                yield token

    def generate_json_plan(self, task: str, available_tools: dict) -> dict:
        """Generate a structured JSON plan for multi-step Celestial tasks."""
        tools_str = "\n".join(f"  - {k}: {v}" for k, v in available_tools.items())

        plan_prompt = (
            f"<start_of_turn>user\n"
            f"You are a task planner. Output ONLY valid JSON, no explanation.\n"
            f"Available tools:\n{tools_str}\n\n"
            f'Format: {{"steps": [{{"tool": "tool_name", "params": {{}}}}, ...]}}\n'
            f'Last step must be: {{"tool": "final_response", "params": {{}}}}\n'
            f"Task: {task}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        raw = self.generate(plan_prompt, max_tokens=400, temperature=0.05)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"steps": [{"tool": "final_response", "params": {}}]}

    @property
    def is_loaded(self) -> bool:
        return self._loaded