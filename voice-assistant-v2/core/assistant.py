import logging
import os
import re
import signal
import sys
import threading
import time
from typing import Callable, Optional

from config.settings import CONFIG
from core.agent import AgentBrain
from core.llm_manager import sanitize_response
from core.tts_stream import StreamingTTS
from memory.memory_manager import MemoryManager
from voice.base import BaseSTT, BaseTTS
from voice.stt import WhisperSTT
from voice.tts import KokoroTTS
from voice.wake_word import OpenWakeWordListener

# Ensure tool registrations run at startup.
import skills  # noqa: F401

logger = logging.getLogger(__name__)


def _print_banner():
    gpu = "GPU" if CONFIG.llm_n_gpu_layers > 0 else "CPU"
    print("\n\033[96m" + "=" * 52)
    print("   ╔═╗╔╗╔╔═╗╔╗╔╔╦╗╦ ╦╔╦╗")
    print("   ╠═╣║║║╠═╣║║║ ║ ║ ║║║║")
    print("   ╩ ╩╝╚╝╩ ╩╝╚╝ ╩ ╚═╝╩ ╩")
    print(f"   Edge AI Voice Assistant  [{gpu} Mode]")
    print("=" * 52 + "\033[0m\n")


def _auto_detect_gpu_layers() -> int:
    """Detect available VRAM and suggest a sensible GPU layer count."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8:
            return 32
        elif vram_gb >= 4:
            return 16
        else:
            return 8
    except Exception:
        return 0


def _gpu_layers_explicitly_set() -> bool:
    """Check if user has explicitly set gpu_layers via CLI or saved settings."""
    return bool(CONFIG._gpu_layers_explicitly_set)


class Anantum:
    tts: BaseTTS
    stt: BaseSTT

    def __init__(self, event_sink: Optional[Callable[[dict], None]] = None):
        _print_banner()
        self._event_sink = event_sink
        self._stop_requested = threading.Event()
        self._interrupt = threading.Event()
        self._assistant_streaming = False
        self._wake_terms = tuple(t.lower() for t in CONFIG.wake_word_terms)
        self._wake_enabled = bool(CONFIG.wake_word_enabled)
        self._wake_listener = None
        self._use_openwakeword = False

        logger.info("[1/5] Initializing voice engine (Kokoro)...")
        self._emit("status", state="starting", label="Initializing voice")
        self.tts = KokoroTTS(voice=CONFIG.kokoro_voice, device=CONFIG.tts_device)
        tts_ok = self.tts.load()
        self.tts_stream = StreamingTTS(self.tts)
        if tts_ok:
            logger.info("Kokoro TTS ready")
            try:
                self.tts.speak("Anantum starting up. Please wait.", blocking=True)
            except Exception as e:
                logger.error("TTS startup message failed: %s", e)
        else:
            logger.warning("Text-only mode (install kokoro for voice)")

        logger.info("[2/5] Loading memory system...")
        self._emit("status", state="starting", label="Loading memory")
        self.memory = MemoryManager()
        logger.info("Memory ready — %d memories loaded", len(self.memory.warm))

        logger.info("[3/5] Loading speech recognition (Whisper)...")
        self._emit("status", state="starting", label="Loading speech recognition")
        self.stt = WhisperSTT(CONFIG.whisper_model, CONFIG.stt_device, CONFIG.stt_compute)
        stt_ok = self.stt.load()
        if stt_ok:
            logger.info("Whisper STT ready")
        else:
            logger.warning("Text input fallback active")

        logger.info("[4/5] Language model loading in background...")
        self._emit("status", state="starting", label="Loading language model")
        from core.llm_manager import LLMManager

        if CONFIG.llm_n_gpu_layers == 40 and not _gpu_layers_explicitly_set():
            detected = _auto_detect_gpu_layers()
            if detected != CONFIG.llm_n_gpu_layers:
                logger.info("Auto-detected GPU layers: %d", detected)
                CONFIG.llm_n_gpu_layers = detected

        self.llm = LLMManager(
            model_path=CONFIG.llm_model,
            n_ctx=CONFIG.llm_n_ctx,
            n_threads=CONFIG.llm_n_threads,
            n_gpu_layers=CONFIG.llm_n_gpu_layers,
        )
        self._llm_ready = threading.Event()
        threading.Thread(target=self._load_llm_bg, daemon=True).start()

        logger.info("[5/5] Starting agent brain...")
        self.brain = AgentBrain(self.llm, self.memory, event_sink=self._event_sink)
        logger.info("Agent brain ready")

        if self._wake_enabled:
            self._wake_listener = OpenWakeWordListener(
                model_paths=CONFIG.wake_word_model_paths,
                threshold=CONFIG.wake_word_threshold,
            )
            self._use_openwakeword = self._wake_listener.start()
            if self._use_openwakeword:
                logger.info("Wake mode: OpenWakeWord active (instant)")
            else:
                logger.info("Wake mode: text fallback active")

        signal.signal(signal.SIGINT, self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

        gpu_str = f"GPU ({CONFIG.llm_n_gpu_layers} layers)" if CONFIG.llm_n_gpu_layers > 0 else "CPU"
        logger.info("[*] Anantum is online  [%s]  - LLM warming up...", gpu_str)
        self._emit("status", state="idle", label="Idle")
        try:
            self.tts.speak("Ready. Instant tools available now. Language model loading in background.", blocking=True)
        except Exception as e:
            logger.error("Failed to speak final startup message: %s", e)

    def _load_llm_bg(self):
        load_thread = threading.Thread(target=self.llm.load, daemon=True)
        load_thread.start()
        load_thread.join(timeout=120)
        if load_thread.is_alive():
            logger.error("LLM load timed out after 120s — running tool-only mode")
            self._emit("error", message="Model load timed out. Check your .gguf file.")
            return
        try:
            self.memory.set_llm(self.llm)
            self._llm_ready.set()
            layers = CONFIG.llm_n_gpu_layers
            status = f"GPU ({layers} layers)" if layers > 0 else "CPU"
            logger.info("Gemma 3 1B ready on %s", status)
            self._emit("status", state="idle", label="Model ready")
            self.tts_stream.push_text("Language model ready. I'm fully operational now.")
        except FileNotFoundError as e:
            logger.error("LLM not found: %s", e)
            logger.info("Running tool-only mode.")
            self._emit("error", message=f"LLM not found: {e}")
        except Exception as e:
            logger.error("LLM load failed: %s", e)
            self._emit("error", message=f"LLM load failed: {e}")

    def _on_user_speaking(self):
        if self._assistant_streaming:
            self._interrupt.set()
            self.tts_stream.stop()
        self._emit("status", state="listening", label="Listening")

    def _contains_wake_term(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in self._wake_terms)

    def _strip_wake_term(self, text: str) -> str:
        stripped = text
        for term in self._wake_terms:
            stripped = re.sub(rf"\b{re.escape(term)}\b", "", stripped, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", stripped).strip(" ,.!?")

    def _respond_streaming(self, user_text: str) -> str:
        sentence_re = re.compile(r"(.+?[.!?])(?:\s+|$)", re.DOTALL)
        stream_state = {"buf": "", "spoken": 0}

        def _emit_chunk(chunk: str):
            clean = sanitize_response(chunk)
            if not clean:
                return
            print(clean + " ", end="", flush=True)
            self._emit("assistant_delta", text=clean + " ")
            self.tts_stream.push_text(clean)
            stream_state["spoken"] += 1

        def on_token(token: str):
            if self._interrupt.is_set():
                return False
            stream_state["buf"] += token
            while True:
                match = sentence_re.match(stream_state["buf"])
                if not match:
                    break
                _emit_chunk(match.group(1))
                stream_state["buf"] = stream_state["buf"][match.end():]
            return True

        self._assistant_streaming = True
        self._interrupt.clear()
        self._emit("status", state="thinking", label="Thinking")
        response = self.brain.respond(user_text, on_token=on_token)

        if not self._interrupt.is_set():
            tail = sanitize_response(stream_state["buf"])
            if tail:
                _emit_chunk(tail)

            clean_response = sanitize_response(response)
            if stream_state["spoken"] == 0 and not tail and clean_response:
                print(clean_response, end="", flush=True)
                self._emit("assistant_delta", text=clean_response)
                self.tts_stream.push_text(clean_response)

        self._assistant_streaming = False
        final = sanitize_response(response)
        self._emit("assistant_final", text=final)
        self._emit("status", state="idle", label="Idle")
        return final

    def process_text_once(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        self._emit("transcript", text=text)
        print(f"\033[93mYou:\033[0m {text}")
        print("\033[94mAnantum:\033[0m ", end="", flush=True)
        response = self._respond_streaming(text)
        if response:
            print()
        print()
        return response

    def run_voice(self):
        logger.info("Voice mode active. Ctrl+C to exit.")
        print("\033[36m[voice mode]  Ctrl+C to exit\033[0m\n")
        self._emit("status", state="idle", label="Idle")

        while not self._stop_requested.is_set():
            try:
                if self._wake_enabled and self._use_openwakeword:
                    self._emit("status", state="idle", label="Waiting wake word")
                    print("\033[90m● Waiting wake word...\033[0m", end="", flush=True)
                    if not self._wake_listener.wait_for_wake(timeout=None):
                        continue
                    self._emit("status", state="listening", label="Listening")
                    print("\r\033[90m● Listening...         \033[0m", end="", flush=True)

                audio_path = self.stt.record(on_speech_start=self._on_user_speaking)
                if not audio_path:
                    continue

                try:
                    text = self.stt.transcribe(audio_path)
                finally:
                    try:
                        os.unlink(audio_path)
                    except OSError:
                        pass

                if not text or len(text.strip()) < 2:
                    continue

                if self._wake_enabled and not self._use_openwakeword and not self._contains_wake_term(text):
                    continue

                if self._wake_enabled and not self._use_openwakeword:
                    text = self._strip_wake_term(text)
                    if not text:
                        self.tts_stream.push_text("Yes?")
                        continue

                self._emit("transcript", text=text)
                print(f"\033[93mYou:\033[0m {text}")
                print("\033[94mAnantum:\033[0m ", end="", flush=True)
                self._emit("status", state="thinking", label="Thinking")

                self._respond_streaming(text)
                print("\n")

            except KeyboardInterrupt:
                break

    def run_text(self):
        logger.info("Text mode active. Type 'exit' to quit.")
        print("\033[36m[text mode]  type 'exit' to quit\033[0m\n")
        self._emit("status", state="idle", label="Idle")

        while not self._stop_requested.is_set():
            try:
                text = input("\033[93mYou:\033[0m ").strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "bye", "goodbye"):
                    self.tts_stream.push_text("Goodbye! Have a great day.")
                    break
                self.process_text_once(text)
            except (KeyboardInterrupt, EOFError):
                break

    def stop(self) -> None:
        if self._stop_requested.is_set():
            return
        logger.info("Stopping assistant runtime")
        self._stop_requested.set()
        self._interrupt.set()
        self.tts_stream.stop()
        if self._wake_listener is not None:
            self._wake_listener.stop()
        self.memory.on_session_end()
        self._emit("status", state="stopped", label="Stopped")

    def _emit(self, event_type: str, **payload) -> None:
        if self._event_sink is None:
            return
        try:
            event = {"type": event_type}
            event.update(payload)
            self._event_sink(event)
        except Exception:
            logger.debug("Failed to emit event '%s'", event_type, exc_info=True)

    def _on_exit(self, *args):
        logger.info("Saving session...")
        print("\n\033[90m[Shutdown] Saving session...\033[0m")
        self.stop()
        time.sleep(0.8)
        print("\033[90m[Shutdown] Goodbye!\033[0m")
        sys.exit(0)
