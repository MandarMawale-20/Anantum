# Build JSON tool-step plans from natural language tasks.

import json
import logging
import re

from core.llm_manager import format_turn, format_generation_start

logger = logging.getLogger(__name__)


class Planner:

    def __init__(self, llm):
        self.llm = llm

    def generate_plan(self, task: str, available_tools: dict) -> dict:
        # Ask the LLM for a JSON step list.
        tools_str = "\n".join(f"  - {k}: {v}" for k, v in available_tools.items())

        plan_prompt = (
            format_turn("user",
                f"You are a task planner. Output ONLY valid JSON, no explanation.\n"
                f"Available tools:\n{tools_str}\n\n"
                f'Format: {{"steps": [{{"tool": "tool_name", "params": {{}}}}, ...]}}\n'
                f'Last step must be: {{"tool": "final_response", "params": {{}}}}\n'
                f"Task: {task}")
            + format_generation_start()
        )

        raw = self.llm.generate(plan_prompt, max_tokens=400, temperature=0.05)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Planner returned invalid JSON; using fallback plan")
        return {"steps": [{"tool": "final_response", "params": {}}]}
