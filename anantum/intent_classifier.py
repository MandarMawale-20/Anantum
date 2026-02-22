# intent_classifier.py — fast pattern-based routing before the LLM
# Handles ~80% of requests in <5ms using regex rules.

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class IntentType(Enum):
    # Fast-path intents: resolved by tools without LLM latency.
    SYSTEM_TIME      = "system_time"
    SYSTEM_DATE      = "system_date"
    TIMER_SET        = "timer_set"
    TIMER_LIST       = "timer_list"
    TIMER_CANCEL     = "timer_cancel"
    NOTE_SAVE        = "note_save"
    NOTE_READ        = "note_read"
    WEATHER          = "weather"
    CALCULATOR       = "calculator"
    DEVICE_INFO      = "device_info"
    MODE_SWITCH      = "mode_switch"
    MEMORY_RECALL    = "memory_recall"
    MEMORY_CLEAR     = "memory_clear"
    GREETING         = "greeting"
    HELP             = "help"
    # Require either LLM or internet or both.
    WEB_SEARCH       = "web_search"
    CONVERSATION     = "conversation"
    CELESTIAL_TASK   = "celestial_task"


@dataclass
class Intent:
    type: IntentType
    confidence: float       # 0.0 - 1.0
    params: dict
    requires_internet: bool = False
    requires_llm: bool = False
    fast_path: bool = True  # True if no LLM needed


# Pattern rules, ordered from most specific to most general.
# Each entry: (IntentType, [regex patterns], {flags})
PATTERNS = [
    # mode switch
    (IntentType.MODE_SWITCH, [
        r"activate celestial mode",
        r"anantum.*celestial",
        r"switch to celestial",
        r"enter celestial",
        r"exit celestial",
        r"disable celestial",
        r"normal mode",
        r"switch to normal",
    ], {"requires_internet": False, "requires_llm": False}),

    # time
    (IntentType.SYSTEM_TIME, [
        r"what(?:'s| is) the time",
        r"current time",
        r"what time is it",
        r"tell me the time",
        r"time (?:right )?now",
        r"what time",
    ], {}),

    # date
    (IntentType.SYSTEM_DATE, [
        r"what(?:'s| is) (?:today(?:'s)? )?date",
        r"what day is (?:it|today)",
        r"today(?:'s)? date",
        r"current date",
        r"what(?:'s| is) today",
    ], {}),

    # timers
    (IntentType.TIMER_SET, [
        r"set (?:a )?timer (?:for )?(.+)",
        r"remind me (?:in|after) (.+)",
        r"alarm (?:for|in) (.+)",
        r"timer (.+)",
        r"after (\d+\s*(?:second|minute|hour|sec|min|hr)s?)",
        r"in (\d+\s*(?:second|minute|hour|sec|min|hr)s?)",
    ], {}),

    (IntentType.TIMER_LIST, [
        r"(?:show|list|what are)(?: my)? timers",
        r"active timers",
        r"any timers",
        r"timers running",
    ], {}),

    (IntentType.TIMER_CANCEL, [
        r"cancel (?:all )?timers?",
        r"stop (?:all )?timers?",
        r"delete timers?",
        r"clear timers?",
    ], {}),

    # notes
    (IntentType.NOTE_SAVE, [
        r"(?:take|make|save|add) a? ?note[:\-]?\s*(.*)",
        r"note (?:that|this)[:\-]?\s*(.*)",
        r"remember (?:that|this)[:\-]?\s*(.*)",
        r"jot (?:this|it) down[:\-]?\s*(.*)",
        r"write (?:this|that) down[:\-]?\s*(.*)",
    ], {}),

    (IntentType.NOTE_READ, [
        r"(?:read|show|list|get)(?: my)? notes?",
        r"what(?:'s| are) (?:in )?my notes",
        r"all notes",
        r"recent notes",
    ], {}),

    # weather (needs internet)
    (IntentType.WEATHER, [
        r"(?:what(?:'s| is) the )?weather(?: in (.+))?",
        r"temperature(?: in (.+))?",
        r"forecast(?: for (.+))?",
        r"(?:is it|will it) (?:rain|sunny|hot|cold|cloudy)(?: in (.+))?",
        r"how(?:'s| is) the weather",
    ], {"requires_internet": True}),

    # calculator
    (IntentType.CALCULATOR, [
        r"(?:what(?:'s| is) |calculate |compute |eval |solve )?([\d\s\+\-\*\/\.\(\)\^%]+(?:[\+\-\*\/][\d\s\+\-\*\/\.\(\)\^%]+)+)",
        r"(\d+)\s*(?:plus|minus|times|divided by|multiplied by)\s*(\d+)",
        r"what(?:'s| is) (\d+) (?:percent|%) of (\d+)",
        r"square root of (\d+)",
        r"(\d+) squared",
    ], {}),

    # device / system info
    (IntentType.DEVICE_INFO, [
        r"(?:cpu|processor) usage",
        r"(?:ram|memory) usage",
        r"(?:disk|storage) (?:space|usage)",
        r"system (?:info|status|stats)",
        r"battery (?:level|status|percentage)",
        r"how much (?:ram|memory|storage|disk)",
        r"device info",
    ], {}),

    # memory
    (IntentType.MEMORY_RECALL, [
        r"what(?:'s| is) my name",
        r"what(?:'s| is) my (.+)",
        r"what do you (?:know|remember) about me",
        r"do you remember (?:my |what )?(.+)",
        r"do you remember me",
        r"recall (?:my )?(.+)",
        r"tell me (?:what you know about me|my (?:name|age|location|profile))",
        r"have i (?:ever )?told you",
        r"my profile",
        r"who am i",
        r"what(?:'s| is) (?:my|the) (.+?) (?:i(?:'ve| have) told you|you know)",
    ], {}),

    (IntentType.MEMORY_CLEAR, [
        r"(?:clear|delete|forget|wipe)(?: all)? (?:my )?memory",
        r"forget (?:everything|me)",
        r"reset memory",
        r"clear all memories",
    ], {}),

    # greetings
    (IntentType.GREETING, [
        r"^(?:hi|hello|hey|good morning|good afternoon|good evening|howdy|sup)[\!\.\?]?$",
        r"^(?:hi|hello|hey) anantum[\!\.\?]?$",
    ], {}),

    # help
    (IntentType.HELP, [
        r"^(?:help|what can you do|capabilities|features|commands?)[\?\!]?$",
        r"how do (?:i|you) work",
        r"what are your (?:features|capabilities|skills)",
    ], {}),

    # web search (needs internet + LLM for summarisation)
    (IntentType.WEB_SEARCH, [
        r"search (?:for |the web for )?(.+)",
        r"look up (.+)",
        r"google (.+)",
        r"find (?:me )?(?:information on |info on )?(.+)",
        r"what happened (?:to|with|in) (.+)",
        r"latest news (?:on|about) (.+)",
        r"who is (.+)",
        r"browse (.+)",
    ], {"requires_internet": True, "requires_llm": True, "fast_path": False}),
]


# --- internet check ---
def check_internet(timeout: float = 2.0) -> bool:
    """
    Tries a few different hosts so a single DNS block doesn't give a false negative.
    """
    import socket
    hosts = [
        ("8.8.8.8", 53),        # Google DNS
        ("1.1.1.1", 53),        # Cloudflare DNS
        ("142.250.80.46", 80),  # google.com HTTP
        ("api.open-meteo.com", 80),  # weather API itself
    ]
    for host, port in hosts:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            continue
    return False


# --- pre-classifier ---

class IntentPreClassifier:
    """Regex-based classifier that runs <1ms. This is the first gate:
    
    ~80% of requests match a fast-path intent and exit immediately.
    Others fall through to the LLM. The speed here buys us responsive UX on common queries.
    """

    def __init__(self):
        # Compile patterns once at startup to avoid re-compiling on every request.
        self._compiled = []
        for intent_type, patterns, flags in PATTERNS:
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            self._compiled.append((intent_type, compiled, flags))

        # Cache internet status to avoid pinging on every request.
        # 30s TTL balances responsiveness vs network chattiness.
        self._internet_available: Optional[bool] = None
        self._internet_checked_at: float = 0
        self._internet_cache_ttl: float = 30

    def classify(self, text: str) -> Intent:
        """Returns an Intent for the given text. Falls back to CONVERSATION if nothing matches."""
        text_stripped = text.strip()
        t0 = time.monotonic()

        for intent_type, compiled_patterns, flags in self._compiled:
            for pattern in compiled_patterns:
                match = pattern.search(text_stripped)
                if match:
                    params = self._extract_params(intent_type, match, text_stripped)
                    confidence = 0.95

                    requires_internet = flags.get("requires_internet", False)
                    requires_llm = flags.get("requires_llm", False)
                    fast_path = flags.get("fast_path", True)

                    if requires_internet:
                        online = self._check_internet_cached()
                        if not online:
                            # return a fast canned response instead of trying the tool
                            return Intent(
                                type=IntentType.CONVERSATION,
                                confidence=0.9,
                                params={"no_internet_for": intent_type.value},
                                requires_internet=True,
                                requires_llm=False,
                                fast_path=True
                            )

                    elapsed = (time.monotonic() - t0) * 1000
                    # print(f"[Intent] {intent_type.value} matched in {elapsed:.1f}ms")
                    return Intent(
                        type=intent_type,
                        confidence=confidence,
                        params=params,
                        requires_internet=requires_internet,
                        requires_llm=requires_llm,
                        fast_path=fast_path
                    )

        # No pattern matched — needs LLM
        return Intent(
            type=IntentType.CONVERSATION,
            confidence=0.7,
            params={},
            requires_internet=False,
            requires_llm=True,
            fast_path=False
        )

    def _extract_params(self, intent_type: IntentType, match: re.Match, text: str) -> dict:
        """Pull relevant parameters out of the regex match groups."""
        groups = [g for g in match.groups() if g is not None]
        params = {}

        if intent_type == IntentType.TIMER_SET:
            duration_str = groups[0] if groups else text
            params["raw"] = duration_str.strip()
            params["seconds"] = self._parse_duration(duration_str)

        elif intent_type == IntentType.NOTE_SAVE:
            params["content"] = groups[0].strip() if groups else text

        elif intent_type == IntentType.MEMORY_RECALL:
            params["topic"] = groups[0].strip() if groups else None
            params["original_query"] = text

        elif intent_type == IntentType.WEATHER:
            params["location"] = groups[0].strip() if groups else None

        elif intent_type == IntentType.WEB_SEARCH:
            params["query"] = groups[0].strip() if groups else text

        elif intent_type == IntentType.CALCULATOR:
            params["expression"] = groups[0] if groups else text
            params["raw_text"] = text

        elif intent_type == IntentType.MODE_SWITCH:
            lower = text.lower()
            if any(w in lower for w in ["celestial", "activate", "enter"]):
                params["mode"] = "celestial"
            else:
                params["mode"] = "normal"

        return params

    def _parse_duration(self, text: str) -> int:
        """Convert a natural language duration string into seconds. Defaults to 60."""
        text = text.lower().strip()
        total = 0
        patterns = [
            (r"(\d+)\s*(?:hour|hr)s?", 3600),
            (r"(\d+)\s*(?:minute|min)s?", 60),
            (r"(\d+)\s*(?:second|sec)s?", 1),
        ]
        for pattern, multiplier in patterns:
            m = re.search(pattern, text)
            if m:
                total += int(m.group(1)) * multiplier
        return total if total > 0 else 60  # default to 1 minute if nothing parsed

    def _check_internet_cached(self) -> bool:
        """Cached connectivity check — re-checks every 30 seconds."""
        now = time.time()
        if self._internet_available is None or (now - self._internet_checked_at) > self._internet_cache_ttl:
            self._internet_available = check_internet()
            self._internet_checked_at = now
        return self._internet_available