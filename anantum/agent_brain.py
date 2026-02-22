# agent_brain.py — routes user input to the right handler
# Normal mode handles fast-path tools and LLM conversation.
# Celestial mode adds web research and multi-step planning.

import re
import time
import threading
from typing import Callable, Optional

from intent_classifier import IntentPreClassifier, IntentType, Intent
from tools import ToolRegistry
from memory_system import MemoryManager
from llm_manager import LLMManager, build_prompt


GREETING_RESPONSES = [
    "Hey, I'm Anantum. What do you need?",
    "Hello! What can I help you with?",
    "Hi! Go ahead.",
]

HELP_TEXT = (
    "Here's what I can do:\n"
    "  [instant]  time, date, timers, notes, calculator, system info, weather\n"
    "  [smart]    conversations, questions, writing, code help\n"
    "  [celestial] web research, multi-step tasks\n"
    "              activate with: 'Anantum, activate Celestial Mode'\n"
    "  [memory]   I remember things you tell me across sessions"
)

NO_INTERNET = {
    "weather":    "No internet connection right now — can't fetch weather. Try again when online.",
    "web_search": "Web search needs internet, which isn't available right now.",
    "default":    "That needs internet, which isn't available right now.",
}

# Patterns that detect LLM garbage: incomplete generation, formatting errors, or filler.
# Raised confidence on these because they're almost always mistakes, not real responses.
GARBAGE_PATTERNS = [
    r"^\[.+\]$",                          # e.g., [intent: none] from instruct leakage
    r"^(none|null|undefined|n/a)$",       # model crashed to placeholder
    r"^<.+>$",                            # HTML/template tag leaked to output
    r"responding to your request",         # classic filler phrase model repeats
    r"^(yes|no|okay|ok|sure|alright)\.$", # one-word answer when user asked real question
]
_GARBAGE_RE = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


def is_garbage(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 3:
        return True
    return any(p.match(t) for p in _GARBAGE_RE)


def clean_response(text: str) -> str:
    """Strip leaked stop tokens and other model artifacts from output."""
    for tok in ["<end_of_turn>", "<start_of_turn>", "[intent:", "[tool:"]:
        if tok in text:
            text = text[:text.index(tok)].strip()
    return text.strip()


# --- Celestial Executor ---
class CelestialExecutor:
    def __init__(self, llm: LLMManager, memory: MemoryManager):
        self.llm = llm
        self.memory = memory

    def execute(self, plan: dict, task: str,
                on_step: Callable[[str, dict], None] = None) -> str:
        steps = plan.get("steps", [])
        results = {}

        for i, step in enumerate(steps):
            tool = step.get("tool")
            params = self._resolve_refs(step.get("params", {}), results)

            if tool == "final_response":
                break

            result = ToolRegistry.run(tool, **params)
            results[f"step_{i}"] = result

            if on_step:
                disp = result.get("result", {})
                if isinstance(disp, dict):
                    disp = disp.get("display", "")
                if disp:
                    on_step(tool, {"display": disp})

        return self._synthesize(task, results)

    def _resolve_refs(self, params: dict, results: dict) -> dict:
        """Replace $step_N references in params with actual step results."""
        out = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$step_"):
                ref = v[1:]
                out[k] = results.get(ref, {}).get("result", v)
            else:
                out[k] = v
        return out

    def _synthesize(self, task: str, results: dict) -> str:
        summaries = []
        for k, r in results.items():
            res = r.get("result", {})
            if isinstance(res, dict):
                summaries.append(res.get("display", str(res))[:300])
            else:
                summaries.append(str(res)[:300])

        prompt = (
            f"<start_of_turn>user\n"
            f"Summarize these tool results into a clear, natural response for: '{task}'\n"
            f"Results:\n" + "\n".join(summaries) +
            f"\n<end_of_turn>\n<start_of_turn>model\n"
        )
        return self.llm.generate(prompt, max_tokens=250)


# --- Agent Brain ---
class AgentBrain:
    def __init__(self, llm: LLMManager, memory: MemoryManager):
        self.llm = llm
        self.memory = memory
        self.classifier = IntentPreClassifier()
        self.executor = CelestialExecutor(llm, memory)
        self._mode = "normal"
        self._greeting_idx = 0

    @property
    def mode(self) -> str:
        return self._mode

    def respond(self, user_input: str, on_token: Callable = None) -> str:
        user_input = user_input.strip()

        self.memory.process_user_message(user_input)

        intent = self.classifier.classify(user_input)

        # Celestial mode intercepts CONVERSATION intents to do multi-step task planning.
        # Fast-path tools exit early and never reach this branch.
        if self._mode == "celestial" and intent.type == IntentType.CONVERSATION:
            response = self._celestial(user_input, on_token)
        else:
            response = self._route(user_input, intent, on_token)

        # Filter garbage before memory: prevents poisoning long-term facts with cruft.
        if response and not is_garbage(response):
            self.memory.process_assistant_message(response)

        return response

    def _route(self, text: str, intent: Intent, cb: Callable) -> str:
        if intent.params.get("no_internet_for"):
            tool_key = intent.params["no_internet_for"]
            msg = NO_INTERNET.get(tool_key, NO_INTERNET["default"])
            self._emit(msg, cb)
            return msg

        # fast-path tool handling (no LLM involved)

        if intent.type == IntentType.SYSTEM_TIME:
            r = ToolRegistry.run("get_time")["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.SYSTEM_DATE:
            r = ToolRegistry.run("get_date")["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.TIMER_SET:
            r = ToolRegistry.run("set_timer",
                                  seconds=intent.params.get("seconds", 60),
                                  label=intent.params.get("raw", "Timer"))["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.TIMER_LIST:
            r = ToolRegistry.run("list_timers")["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.TIMER_CANCEL:
            r = ToolRegistry.run("cancel_timer")["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.NOTE_SAVE:
            content = intent.params.get("content", text)
            r = ToolRegistry.run("save_note", content=content)["result"]["display"]
            self.memory.store_fact(f"User note: {content}", importance=0.85, memory_type="note")
            self._emit(r, cb); return r

        if intent.type == IntentType.NOTE_READ:
            r = ToolRegistry.run("get_notes", limit=5)["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.WEATHER:
            r = ToolRegistry.run("get_weather",
                                  location=intent.params.get("location"))["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.CALCULATOR:
            r = ToolRegistry.run("calculate",
                                  expression=intent.params.get("expression", ""),
                                  raw_text=intent.params.get("raw_text", text))["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.DEVICE_INFO:
            r = ToolRegistry.run("get_device_info")["result"]["display"]
            self._emit(r, cb); return r

        if intent.type == IntentType.MEMORY_RECALL:
            return self._handle_memory_recall(text, intent, cb)

        if intent.type == IntentType.MEMORY_CLEAR:
            self.memory.warm.entries.clear()
            self.memory.warm._rebuild_index()
            self.memory.warm._save()
            self.memory.hot.clear()
            msg = "Memory cleared. Starting fresh."
            self._emit(msg, cb); return msg

        if intent.type == IntentType.MODE_SWITCH:
            return self._switch_mode(intent.params.get("mode", "normal"), cb)

        if intent.type == IntentType.GREETING:
            msg = GREETING_RESPONSES[self._greeting_idx % len(GREETING_RESPONSES)]
            self._greeting_idx += 1
            self._emit(msg, cb); return msg

        if intent.type == IntentType.HELP:
            self._emit(HELP_TEXT, cb); return HELP_TEXT

        # fall through to LLM for anything we couldn't classify
        return self._converse(text, cb)

    def _handle_memory_recall(self, text: str, intent, cb: Callable) -> str:
        """
        Answers memory questions using the LLM when possible.
        Falls back to a plain-text fact list if the LLM isn't loaded yet.
        """
        topic = intent.params.get("topic")
        query = intent.params.get("original_query", text)

        ctx = self.memory.get_context_for_prompt(query)
        facts = ctx.get("relevant_facts", [])

        if not facts:
            msg = "I don't have anything stored about you yet. Tell me things like your name, where you're from, or what you're working on, and I'll remember them."
            self._emit(msg, cb)
            return msg

        # build a prompt focused on answering the specific question, not a data dump
        facts_text = "\n".join(f"- {f['text']}" for f in facts[:6])

        if self.llm.is_loaded:
            recall_prompt = (
                f"<start_of_turn>user\n"
                f"Based ONLY on these stored facts about the user, answer their question naturally in 1-2 sentences.\n"
                f"Do NOT add any information not in the facts. Do NOT say 'Based on my memory'.\n"
                f"Just answer directly and naturally.\n\n"
                f"Stored facts:\n{facts_text}\n\n"
                f"User question: {query}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            response = self.llm.generate(recall_prompt, max_tokens=100, temperature=0.3)
            response = clean_response(response)
            if response and not is_garbage(response):
                self._emit(response, cb)
                return response

        # fallback: build a readable sentence without format tags
        clean_facts = []
        for f in facts[:4]:
            fact_text = f["text"]
            import re
            fact_text = re.sub(r"^\[[\w_]+\]\s*", "", fact_text)  # strip [type] prefix
            clean_facts.append(fact_text)

        msg = "Here's what I know about you: " + ", ".join(clean_facts) + "."
        self._emit(msg, cb)
        return msg

    def _converse(self, text: str, cb: Callable) -> str:
        if not self.llm.is_loaded:
            msg = "Still loading the language model, give me a moment..."
            self._emit(msg, cb)
            return msg

        # short/vague inputs cause hallucinations, ask for clarification instead
        words = text.strip().split()
        if len(words) <= 2 and len(text.strip()) < 12:
            msg = "Could you say that again? I didn't quite catch it."
            self._emit(msg, cb)
            return msg

        ctx = self.memory.get_context_for_prompt(text)
        prompt = build_prompt(
            user_message=text,
            memory_context=ctx,
            conversation_history=ctx["recent_turns"],
        )

        if cb:
            return self._stream_sentences(prompt, cb)
        else:
            raw = self.llm.generate(prompt)
            return clean_response(raw)

    def _celestial(self, text: str, cb: Callable) -> str:
        if not self.llm.is_loaded:
            return "Language model not ready yet."

        self._emit("Planning task...", cb)
        tools = ToolRegistry.list_tools(mode="celestial")
        plan = self.llm.generate_json_plan(text, tools)

        def on_step(tool_name, result):
            disp = result.get("display", "")
            if disp and cb:
                cb(f"[{tool_name}]: {disp}")

        response = self.executor.execute(plan, text, on_step)
        response = clean_response(response)
        if cb:
            cb(response)
        return response

    def _switch_mode(self, mode: str, cb: Callable) -> str:
        self._mode = mode
        if mode == "celestial":
            msg = "Celestial Mode activated. I can now handle multi-step tasks and web research."
        else:
            msg = "Switched back to Normal Mode."
        self._emit(msg, cb)
        return msg

    def _stream_sentences(self, prompt: str, cb: Callable) -> str:
        """Stream tokens and emit sentence-by-sentence to TTS for lower latency.
        
        This unlocks real-time interaction: TTS starts speaking first sentence
        while LLM still generating rest. Without streaming, user waits for full response.
        """
        sentence_end = {'.', '!', '?'}
        buffer = ""
        full = ""

        for token in self.llm.generate_streaming(prompt, max_tokens=200):
            # Stop token leaked to output; clean exit rather than partial response.
            if any(s in token for s in ["<end_of_turn>", "<start_of_turn>"]):
                break

            buffer += token
            full += token

            if buffer.rstrip() and buffer.rstrip()[-1] in sentence_end:
                sentence = clean_response(buffer)
                if sentence and not is_garbage(sentence):
                    cb(sentence)
                buffer = ""

        # flush whatever's left in the buffer
        if buffer.strip():
            sentence = clean_response(buffer)
            if sentence and not is_garbage(sentence):
                cb(sentence)

        return clean_response(full)

    @staticmethod
    def _emit(text: str, cb: Callable = None):
        if cb and text and not is_garbage(text):
            cb(text)