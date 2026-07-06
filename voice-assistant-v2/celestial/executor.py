# Execute planner steps and synthesize final response.

import logging
from typing import Callable

from core.llm_manager import format_turn, format_generation_start
from skills.base import ToolRegistry

logger = logging.getLogger(__name__)


class CelestialExecutor:
    def __init__(self, llm, memory):
        self.llm    = llm
        self.memory = memory

    def execute(self, plan: dict, task: str,
                on_step: Callable[[str, dict], None] = None) -> str:
        # Run each step in order and collect tool outputs.
        results = {}
        steps = plan.get("steps", [])

        for i, step in enumerate(steps):
            tool = step.get("tool")
            params = self._resolve_refs(step.get("params", {}), results)

            if tool == "final_response":
                break

            result = ToolRegistry.run(tool, **params)
            results[f"step_{i}"] = result

            if not result.get("success", True):
                logger.warning("Step %d tool '%s' failed: %s", i, tool, result.get("error"))

            if on_step:
                disp = result.get("result", {})
                if isinstance(disp, dict):
                    disp = disp.get("display", "")
                if disp:
                    on_step(tool, {"display": disp})

        return self._synthesize(task, results)

    def _resolve_refs(self, params: dict, results: dict) -> dict:
        # Resolve $step_N references.
        out = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$step_"):
                ref = v[1:]
                out[k] = results.get(ref, {}).get("result", v)
            else:
                out[k] = v
        return out

    def _synthesize(self, task: str, results: dict) -> str:
        """Ask the LLM to turn raw tool outputs into a natural language response."""
        summaries = []
        for k, r in results.items():
            if not r.get("success", True):
                summaries.append(f"[Error in {k}: {r.get('error', 'unknown')}]")
                continue
            res = r.get("result", {})
            if isinstance(res, dict):
                summaries.append(res.get("display", str(res))[:300])
            else:
                summaries.append(str(res)[:300])

        prompt = (
            format_turn("user",
                f"Summarize these tool results into a clear, natural response for: '{task}'\n"
                f"Results:\n" + "\n".join(summaries))
            + format_generation_start()
        )
        return self.llm.generate(prompt, max_tokens=250)
