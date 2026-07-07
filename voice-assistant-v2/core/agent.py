import logging
import random
from pathlib import Path
from typing import Callable, Optional

from core.context_builder import ContextBuilder
from core.intent_detector import IntentPreClassifier, IntentType
from core.llm_manager import LLMManager, format_generation_start, format_turn
from memory.memory_manager import MemoryManager
from skills.base import ToolRegistry

logger = logging.getLogger(__name__)


def _load_system_prompt(mode: str = "normal") -> str:
    prompt_name = "celestial_prompt.txt" if mode == "celestial" else "system_prompt.txt"
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_name

    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to load %s: %s, using default", prompt_name, exc)

    if mode == "celestial":
        return (
            "You are Anantum in CELESTIAL MODE - a voice-based AI assistant with real-time web search. "
            "Keep answers brief unless asked for detail. Use web search for current events and data. "
            "Be friendly and conversational. Never hallucinate — if unsure, say so."
        )

    return (
        "You are Anantum, a helpful, witty, and concise offline voice assistant. "
        "You run entirely on-device with no data leaving the machine. "
        "Keep answers brief unless the user asks for detail. "
        "When uncertain, say so honestly — never hallucinate facts."
    )


class AgentBrain:
    """Routes user requests to tools or the LLM; manages celestial mode state."""

    def __init__(
        self,
        llm: LLMManager,
        memory: MemoryManager,
        event_sink: Optional[Callable[[dict], None]] = None,
    ):
        self.llm = llm
        self.memory = memory
        self._event_sink = event_sink
        self._classifier = IntentPreClassifier()
        self._ctx_builder = ContextBuilder(memory)
        self._mode = "normal"

        self._dispatch: dict[IntentType, Callable[[dict], str]] = {
            IntentType.SYSTEM_TIME: self._handle_time,
            IntentType.SYSTEM_DATE: self._handle_date,
            IntentType.TIMER_SET: self._handle_timer_set,
            IntentType.TIMER_LIST: self._handle_timer_list,
            IntentType.TIMER_CANCEL: self._handle_timer_cancel,
            IntentType.NOTE_SAVE: self._handle_note_save,
            IntentType.NOTE_READ: self._handle_note_read,
            IntentType.WEATHER: self._handle_weather,
            IntentType.CALCULATOR: self._handle_calculator,
            IntentType.DEVICE_INFO: self._handle_device_info,
            IntentType.MEMORY_RECALL: self._handle_memory_recall,
            IntentType.MEMORY_CLEAR: self._handle_memory_clear,
            IntentType.MODE_SWITCH: self._handle_mode_switch,
            IntentType.GREETING: self._handle_greeting,
            IntentType.HELP: self._handle_help,
            IntentType.WEB_SEARCH: self._handle_web_search,
        }

    def respond(
        self,
        user_message: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        self.memory.process_user_message(user_message)
        context = self._ctx_builder.build(user_message)
        intent = self._classifier.classify(user_message)
        logger.debug("Intent: %s (confidence=%.2f)", intent.type.value, intent.confidence)

        response = self._route(intent, user_message, context, on_token)
        self.memory.process_assistant_message(response)
        return response

    def _route(
        self,
        intent,
        user_message: str,
        context: dict,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        no_internet_for = intent.params.get("no_internet_for")
        if no_internet_for:
            return f"I'd need internet access for that ({no_internet_for}), but I'm currently offline."

        handler = self._dispatch.get(intent.type)
        if handler:
            return handler(intent.params)
        return self._converse(user_message, context, on_token)

    def _safe_tool_run(self, tool_name: str, **kwargs) -> dict:
        result = ToolRegistry.run(tool_name, **kwargs)
        if result is None:
            result = {}
        normalized = result.get("result")
        if normalized is None:
            result = {k: v for k, v in result.items() if k != "result"}
        payload = {
            "tool": tool_name,
            "success": bool(result.get("success", True)),
            "result": result.get("result") or {},
            "error": result.get("error"),
        }
        if isinstance(payload["result"], dict):
            payload["display"] = payload["result"].get("display")
        self._emit("tool_result", **payload)
        if not result.get("success", True):
            logger.warning("Tool '%s' failed: %s", tool_name, result.get("error"))
        return result

    def _emit(self, event_type: str, **payload) -> None:
        if self._event_sink is None:
            return
        try:
            event = {"type": event_type}
            event.update(payload)
            self._event_sink(event)
        except Exception:
            logger.debug("Failed to emit event '%s'", event_type, exc_info=True)

    def _converse(self, user_message: str, context: dict, on_token: Optional[Callable[[str], None]]) -> str:
        if self.llm.llm is None:
            return (
                "I'm still loading my language model. "
                "Instant tool commands (time, notes, timers) work right now."
            )

        system_prompt = self._build_system_prompt(context)
        prompt = format_turn("system", system_prompt) + self.llm.build_prompt(
            user_message,
            chat_history=context.get("recent_turns", []),
            past_summaries=self._join_past_summaries(context),
        )
        return self.llm.generate(prompt, on_token=on_token)

    def _build_system_prompt(self, context: dict) -> str:
        system_prompt = _load_system_prompt(self._mode)
        relevant_facts = context.get("relevant_facts", [])
        if relevant_facts:
            facts_text = "\n".join(f"- {entry['text']}" for entry in relevant_facts)
            system_prompt += f"\n\n[Known facts about the user]\n{facts_text}"
        return system_prompt

    def _join_past_summaries(self, context: dict) -> Optional[str]:
        past_summaries = context.get("past_summaries", []) or []
        if not past_summaries:
            return None
        return "\n".join(past_summaries)

    def _handle_time(self, params: dict) -> str:
        result = self._safe_tool_run("get_time")
        return result.get("result", {}).get("display", "Couldn't read the time.")

    def _handle_date(self, params: dict) -> str:
        result = self._safe_tool_run("get_date")
        return result.get("result", {}).get("display", "Couldn't read the date.")

    def _handle_timer_set(self, params: dict) -> str:
        result = self._safe_tool_run(
            "set_timer",
            seconds=params.get("seconds", 60),
            label=params.get("raw", "Timer"),
        )
        return result.get("result", {}).get("display", "Timer set.")

    def _handle_timer_list(self, params: dict) -> str:
        result = self._safe_tool_run("list_timers")
        return result.get("result", {}).get("display", "No active timers.")

    def _handle_timer_cancel(self, params: dict) -> str:
        result = self._safe_tool_run("cancel_timer", timer_id=params.get("timer_id"))
        return result.get("result", {}).get("display", "Timers cleared.")

    def _handle_note_save(self, params: dict) -> str:
        result = self._safe_tool_run("save_note", content=params.get("content", ""))
        return result.get("result", {}).get("display", "Note saved.")

    def _handle_note_read(self, params: dict) -> str:
        result = self._safe_tool_run("get_notes")
        return result.get("result", {}).get("display", "No notes found.")

    def _handle_weather(self, params: dict) -> str:
        result = self._safe_tool_run("get_weather", location=params.get("location"))
        return result.get("result", {}).get("display", "Couldn't fetch weather.")

    def _handle_calculator(self, params: dict) -> str:
        result = self._safe_tool_run(
            "calculate",
            expression=params.get("expression", ""),
            raw_text=params.get("raw_text", ""),
        )
        return result.get("result", {}).get("display", "Couldn't compute that.")

    def _handle_device_info(self, params: dict) -> str:
        result = self._safe_tool_run("get_device_info")
        return result.get("result", {}).get("display", "Couldn't read system info.")

    def _handle_memory_recall(self, params: dict) -> str:
        topic = params.get("topic") or params.get("original_query", "")
        warm_results = self.memory.warm.search(topic, top_k=5)
        if not warm_results:
            return "I don't have specific memories about that yet."

        memories_text = "\n".join(f"- {entry.text}" for entry, _ in warm_results)
        prompt = format_turn(
            "user",
            f"The user asked: '{topic}'\n\nBased on these stored memories, answer naturally:\n{memories_text}",
        ) + format_generation_start()
        return self.llm.generate(prompt, max_tokens=200)

    def _handle_memory_clear(self, params: dict) -> str:
        self.memory.clear_all()
        return "Done — I've wiped everything I know about you. Fresh start."

    def _handle_mode_switch(self, params: dict) -> str:
        self._mode = params.get("mode", "normal")
        if self._mode == "celestial":
            return "Celestial mode active. I can now plan multi-step tasks and use web search."
        return "Back to normal mode."

    def _handle_greeting(self, params: dict) -> str:
        return random.choice([
            "Hey! What can I help you with?",
            "Hello! What's on your mind?",
            "Hi there! Ready to help.",
        ])

    def _handle_help(self, params: dict) -> str:
        return (
            "Here's what I can do:\n"
            "  Time & date — 'what time is it?'\n"
            "  Timers      — 'set a timer for 10 minutes'\n"
            "  Notes       — 'save a note: buy milk'\n"
            "  Weather     — 'weather in London'\n"
            "  Calculator  — '15% of 200'\n"
            "  Device info — 'cpu usage'\n"
            "  Memory      — 'do you remember my name?'\n"
            "  Chat        — anything else goes to the LLM\n"
            "  Celestial   — 'activate celestial mode' for multi-step tasks"
        )

    def _handle_web_search(self, params: dict) -> str:
        if self._mode != "celestial":
            return "Web search is only available in celestial mode. Activate with 'activate celestial mode'."
        result = self._safe_tool_run("web_search", query=params.get("query", ""))
        if result.get("success"):
            return result.get("result", {}).get("display", "No results found.")
        return "Web search failed. Check your connection."
