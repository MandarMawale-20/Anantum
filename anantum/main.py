# main.py — Anantum entry point
# Startup order: TTS -> Memory -> STT -> LLM (background) -> Brain

import os
import sys
import time
import queue
import signal
import threading
import tempfile
import wave
from pathlib import Path

# Configuration: edit these for your hardware setup.
# Defaults are tuned for GTX 1650 4GB + 4-core CPU.
CONFIG = {
    "llm_model":           "models/gemma3-voice-Q5_K_M.gguf",
    "whisper_model":       "ctranslate2-4you/distil-whisper-small.en-ct2-float32",
    "kokoro_voice":        "af_bella",   # voices: af_bella, af_sky, am_adam, af_nicole

    # n_gpu_layers: balance between speed and VRAM. 0=CPU only, 35=full GPU for 1650.
    # Q5_K_M is ~850MB; fits in VRAM leaving room for KV cache.
    "llm_n_gpu_layers":    35,
    "llm_n_ctx":           2048,
    "llm_n_threads":       4,

    "stt_device":          "cpu",
    "stt_compute":         "int8",

    # Silence detection: adjust silence_threshold if mic environment is very noisy
    "silence_threshold":   400,
    "silence_duration":    1.8,
    "min_speech_chunks":   8,
    "max_record_seconds":  30,
    "sample_rate":         16000,
    "min_response_length": 3,
}


# TTS (Kokoro) loads first so we can narrate startup progress.
    # Saves time vs waiting for LLM to load to give user feedback.
class KokoroTTS:

    GARBAGE = {"none", "null", "undefined", "n/a", "okay.", "ok.", "yes.", "no."}

    def __init__(self, voice: str = "af_bella", device: str = "auto"):
        self.voice = voice
        self.device = device   # "auto", "cuda", "cpu"
        self._pipe = None
        self._available = False
        self._speak_lock = threading.Lock()
        self._sr = 24000       # Kokoro native sample rate

    def load(self) -> bool:
        try:
            import torch
            from kokoro import KPipeline

            if self.device == "auto":
                use_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                use_device = self.device

            if use_device == "cuda":
                torch.set_default_device("cuda")
                print(f"[TTS] Kokoro loading on GPU (CUDA)")
            else:
                print(f"[TTS] Kokoro loading on CPU")

            self._pipe = KPipeline(lang_code='a')
            self._available = True
            self._gpu = (use_device == "cuda")
            print(f"[TTS] Kokoro ready ({'GPU' if self._gpu else 'CPU'})")
            return True

        except ImportError:
            print("[TTS] kokoro not installed — pip install kokoro sounddevice")
            return False
        except Exception as e:
            print(f"[TTS] Failed to load: {e}")
            return False

    def _should_skip(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        t = text.strip()
        if len(t) < CONFIG["min_response_length"]:
            return True
        if t.startswith("[") and t.endswith("]"):
            return True  # leaked internal tags like [intent: none]
        if t.lower().rstrip(".,!?") in self.GARBAGE:
            return True
        return False

    def speak(self, text: str, blocking: bool = True):
        if self._should_skip(text):
            return
        text = text.strip()

        if not self._available or not self._pipe:
            print(f"\033[94mAnantum:\033[0m {text}")
            return

        try:
            import sounddevice as sd
            import numpy as np

            with self._speak_lock:
                all_audio = []
                # kokoro splits sentences internally and returns chunks
                for _, _, audio in self._pipe(text, voice=self.voice):
                    if audio is not None and len(audio) > 0:
                        all_audio.append(audio)

                if all_audio:
                    combined = np.concatenate(all_audio)
                    sd.play(combined, samplerate=self._sr)
                    if blocking:
                        sd.wait()

        except Exception as e:
            print(f"[TTS Error]: {e}")
            print(f"\033[94mAnantum:\033[0m {text}")

    def speak_nonblocking(self, text: str):
        """Queue speech in a background thread without blocking the caller."""
        if self._should_skip(text):
            return
        threading.Thread(target=self.speak, args=(text, True), daemon=True).start()

    @property
    def available(self) -> bool:
        return self._available


    # STT (Whisper).
    # distil-whisper-small is faster than fullsize with <1% accuracy loss.
class WhisperSTT:

    # Common Whisper hallucinations. Model repeats these when confidence is low.
    _HALLUCINATION_PHRASES = {
        # "Thank you for watching" is the #1 false positive in the wild
        "thank you", "thank you very much", "thank you so much",
        "thanks", "thanks for watching", "thank you for watching",
        "thank you for listening", "thanks for listening",
        "thank you for having me", "thank you for having us",
        "thanks for having me",
        # Subscribe/channel noise
        "please subscribe", "like and subscribe", "don't forget to subscribe",
        "hit the like button", "leave a comment",
        # Common noise misdetections
        "you", "the", ".", "", " ", "i", "a",
        "bye", "goodbye", "see you", "see you next time", "see you later",
        "uh", "um", "uh huh", "mm", "hmm", "ah", "oh", "yeah",
        "okay", "ok", "alright", "right", "sure",
        # Repeated filler
        "yeah yeah yeah", "no no no", "okay okay",
    }

    def __init__(self, model="distil-whisper/distil-small.en", device="cpu", compute="int8"):
        self._model = None
        self._model_name = model
        self._device = device
        self._compute = compute

    def load(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading {self._model_name}...")
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute,
            )
            return True
        except ImportError:
            print("[STT] faster-whisper not installed: pip install faster-whisper")
            return False
        except Exception as e:
            print(f"[STT] Failed to load {self._model_name}: {e}")
            print("[STT] Falling back to tiny.en...")
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel("tiny.en", device=self._device, compute_type=self._compute)
                print("[STT] Loaded tiny.en as fallback")
                return True
            except Exception as e2:
                print(f"[STT] Fallback also failed: {e2}")
                return False

    def _is_hallucination(self, text: str) -> bool:
        if not text:
            return True
        t = text.lower().strip().rstrip(".,!?")
        if t in self._HALLUCINATION_PHRASES:
            return True
        # catch repetitive junk like "thank you thank you thank you"
        words = t.split()
        if len(words) >= 4:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if t.count(phrase) >= 3:
                    return True
            from collections import Counter
            freq = Counter(words)
            top_word, top_count = freq.most_common(1)[0]
            if top_count / len(words) > 0.5 and len(words) > 4:
                return True
        return False

    def transcribe(self, audio_path: str) -> str:
        if not self._model:
            return input("You (text): ").strip()
        try:
            import os
            file_size = os.path.getsize(audio_path)
            # Skip very short audio that's likely noise, not speech.
            # Threshold: 16000 bytes = 0.5s @ 16 kHz 16-bit mono.
            if file_size < 16000:
                return ""

            segments, info = self._model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 400,
                    "speech_pad_ms": 200,
                    "threshold": 0.45,
                },
                condition_on_previous_text=False,  # disables cross-segment history chaining
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=1.9,   # tighter filter catches repeating junk
                temperature=0.0,                   # greedy: avoid sampling noise on short audio
            )

            good_segments = []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                # Skip high no_speech probability
                if hasattr(seg, "no_speech_prob") and seg.no_speech_prob > 0.55:
                    print(f"[STT] Rejected (no_speech={seg.no_speech_prob:.2f}): {text!r}")
                    continue
                if self._is_hallucination(text):
                    print(f"[STT] Rejected (hallucination): {text!r}")
                    continue
                good_segments.append(text)

            result = " ".join(good_segments).strip()

            # Final pass: catch hallucinations spanning multiple segments.
            # E.g., "thank you thank you" sometimes passes segment filtering.
            if self._is_hallucination(result):
                print(f"[STT] Rejected (final filter): {result!r}")
                return ""

            return result
        except Exception as e:
            print(f"[STT Error]: {e}")
            return ""

    def record(self):
        try:
            import sounddevice as sd
            import numpy as np

            sr = CONFIG["sample_rate"]
            chunk_size = int(sr * 0.1)
            threshold = CONFIG["silence_threshold"]
            max_silent = int(CONFIG["silence_duration"] / 0.1)
            min_chunks = CONFIG["min_speech_chunks"]
            max_chunks = int(CONFIG["max_record_seconds"] / 0.1)

            print("\033[90m[mic] listening...\033[0m", end="", flush=True)

            recorded = []
            silent_count = 0
            total = 0
            speech_started = False

            with sd.InputStream(samplerate=sr, channels=1,
                                dtype='int16', blocksize=chunk_size) as stream:
                while total < max_chunks:
                    chunk, _ = stream.read(chunk_size)
                    flat = chunk.flatten()
                    rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))

                    if rms > threshold:
                        if not speech_started:
                            print(" \033[92m●\033[0m", end="", flush=True)
                            speech_started = True
                        silent_count = 0
                        recorded.append(flat)
                    else:
                        if speech_started:
                            recorded.append(flat)
                        silent_count += 1
                        if speech_started and silent_count >= max_silent and total >= min_chunks:
                            break
                    total += 1

            print()

            if not recorded or not speech_started:
                return None

            # Require at least ~0.4 seconds of actual speech
            if len(recorded) < 4:
                return None

            audio = np.concatenate(recorded)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio.tobytes())
            return tmp.name

        except Exception as e:
            print(f"\n[Record Error]: {e}")
            return None


# --- startup banner ---
def print_banner():
    gpu = "GPU" if CONFIG["llm_n_gpu_layers"] > 0 else "CPU"
    print("\n\033[96m" + "═" * 52)
    print("   ╔═╗╔╗╔╔═╗╔╗╔╔╦╗╦ ╦╔╦╗")
    print("   ╠═╣║║║╠═╣║║║ ║ ║ ║║║║")
    print("   ╩ ╩╝╚╝╩ ╩╝╚╝ ╩ ╚═╝╩ ╩")
    print(f"   Edge AI Voice Assistant  [{gpu} Mode]")
    print("═" * 52 + "\033[0m\n")


# --- main assistant ---
class Anantum:
    def __init__(self):
        print_banner()

        # 1. TTS first — so it can narrate the rest of startup
        print("\033[33m[1/5] Initializing voice engine (Kokoro)...\033[0m")
        self.tts = KokoroTTS(voice=CONFIG["kokoro_voice"], device=CONFIG.get("tts_device", "auto"))
        tts_ok = self.tts.load()
        if tts_ok:
            print("\033[32m      ✓ Kokoro TTS ready\033[0m")
            self.tts.speak("Anantum starting up. Please wait.")
        else:
            print("\033[33m      ! Text-only mode (install kokoro for voice)\033[0m")

        # 2. Memory
        print("\033[33m[2/5] Loading memory system...\033[0m")
        from memory_system import MemoryManager
        self.memory = MemoryManager()
        count = len(self.memory.warm)
        print(f"\033[32m      ✓ Memory ready  ({count} memories loaded)\033[0m")

        # 3. STT
        print("\033[33m[3/5] Loading speech recognition (Whisper)...\033[0m")
        self.stt = WhisperSTT(CONFIG["whisper_model"],
                               CONFIG["stt_device"],
                               CONFIG["stt_compute"])
        stt_ok = self.stt.load()
        print("\033[32m      ✓ Whisper STT ready\033[0m" if stt_ok
              else "\033[33m      ! Text input fallback active\033[0m")

        # 4. LLM — background thread so we don't block
        print("\033[33m[4/5] Language model loading in background...\033[0m")
        from llm_manager import LLMManager
        self.llm = LLMManager(
            model_path=CONFIG["llm_model"],
            n_ctx=CONFIG["llm_n_ctx"],
            n_threads=CONFIG["llm_n_threads"],
            n_gpu_layers=CONFIG["llm_n_gpu_layers"],
        )
        self._llm_ready = threading.Event()
        threading.Thread(target=self._load_llm_bg, daemon=True).start()

        # 5. Brain
        print("\033[33m[5/5] Starting agent brain...\033[0m")
        from agent_brain import AgentBrain
        self.brain = AgentBrain(self.llm, self.memory)
        print("\033[32m      ✓ Agent brain ready\033[0m")

        signal.signal(signal.SIGINT, self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

        gpu_str = f"GPU ({CONFIG['llm_n_gpu_layers']} layers)" if CONFIG["llm_n_gpu_layers"] > 0 else "CPU"
        print(f"\n\033[92m[*] Anantum is online  [{gpu_str}]  — LLM warming up...\033[0m\n")
        self.tts.speak("Ready. Instant tools available now. Language model loading in background.")

    def _load_llm_bg(self):
        try:
            self.llm.load()
            self._llm_ready.set()
            layers = CONFIG["llm_n_gpu_layers"]
            status = f"GPU ({layers} layers)" if layers > 0 else "CPU"
            print(f"\n\033[92m[LLM] ✓ Gemma 3 1B ready on {status}\033[0m")
            self.tts.speak("Language model ready. I'm fully operational now.")
        except FileNotFoundError as e:
            print(f"\n\033[91m[LLM] ✗ Not found: {e}\033[0m")
            print("[LLM]   Running tool-only mode.")
        except Exception as e:
            print(f"\n\033[91m[LLM] ✗ {e}\033[0m")

    def run_voice(self):
        print("\033[36m[voice mode]  Ctrl+C to exit\033[0m\n")

        tts_queue = queue.Queue()

        def tts_worker():
            while True:
                item = tts_queue.get()
                if item is None:
                    break
                self.tts.speak(item)
                tts_queue.task_done()

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        def on_sentence(sentence: str):
            if not sentence or not sentence.strip():
                return
            s = sentence.strip()
            if s.startswith("[") and s.endswith("]"):
                return  # suppress internal tags
            print(f"\033[94mAnantum:\033[0m {s}")
            tts_queue.put(s)

        while True:
            try:
                audio_path = self.stt.record()
                if not audio_path:
                    continue

                text = self.stt.transcribe(audio_path)
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

                if not text or len(text.strip()) < 2:
                    continue

                print(f"\033[93mYou:\033[0m {text}")
                self.brain.respond(text, on_token=on_sentence)
                print()

            except KeyboardInterrupt:
                break

        tts_queue.put(None)
        tts_thread.join(timeout=3)

    def run_text(self):
        print("\033[36m[text mode]  type 'exit' to quit\033[0m\n")

        def on_sentence(sentence: str):
            if not sentence or not sentence.strip():
                return
            s = sentence.strip()
            if s.startswith("[") and s.endswith("]"):
                return
            print(f"\033[94mAnantum:\033[0m {s}")
            self.tts.speak(s)

        while True:
            try:
                text = input("\033[93mYou:\033[0m ").strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit", "bye", "goodbye"):
                    self.tts.speak("Goodbye! Have a great day.")
                    break
                self.brain.respond(text, on_token=on_sentence)
                print()
            except (KeyboardInterrupt, EOFError):
                break

    def _on_exit(self, *args):
        print("\n\033[90m[Shutdown] Saving session...\033[0m")
        self.memory.on_session_end()  # triggers final summary in background
        time.sleep(0.8)  # let background thread finish before exit
        print("\033[90m[Shutdown] Goodbye!\033[0m")
        sys.exit(0)


# --- entry point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anantum AI Assistant")
    parser.add_argument("--mode",  choices=["voice", "text"], default="voice")
    parser.add_argument("--model", type=str, help="Path to GGUF model")
    parser.add_argument("--gpu",   type=int, default=None,
                        help="GPU layers to offload (0=CPU only, 35=full GPU for GTX 1650)")
    parser.add_argument("--voice", type=str, default=None,
                        help="Kokoro voice variant: af_bella, af_sky, am_adam, af_nicole")
    parser.add_argument("--tts-device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="TTS device: auto (detect GPU), cuda, cpu")
    args = parser.parse_args()

    if args.model:
        CONFIG["llm_model"] = args.model
    if args.gpu is not None:
        CONFIG["llm_n_gpu_layers"] = args.gpu
    if args.voice:
        CONFIG["kokoro_voice"] = args.voice
    if hasattr(args, "tts_device"):
        CONFIG["tts_device"] = args.tts_device

    assistant = Anantum()

    if args.mode == "text":
        assistant.run_text()
    else:
        assistant.run_voice()