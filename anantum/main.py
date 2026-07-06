"""Anantum — Edge AI Voice Assistant.

Monolithic runtime that orchestrates TTS, STT, LLM, memory, and tool routing
into a single voice/text interactive session.
"""

import os
import sys
import time
import queue
import signal
import threading
import tempfile
import wave

# Runtime configuration — override via CLI flags or .env
CONFIG = {
    "llm_model":           "models/gemma3-voice-Q5_K_M.gguf",
    "whisper_model":       "ctranslate2-4you/distil-whisper-small.en-ct2-float32",
    "kokoro_voice":        "af_bella",
    "llm_n_gpu_layers":    35,
    "llm_n_ctx":           2048,
    "llm_n_threads":       4,
    "stt_device":          "cpu",
    "stt_compute":         "int8",
    "silence_threshold":   400,
    "silence_duration":    1.8,
    "min_speech_chunks":   8,
    "max_record_seconds":  30,
    "sample_rate":         16000,
    "min_response_length": 3,
}


class KokoroTTS:
    """Text-to-speech via Kokoro ONNX pipeline with GPU/CPU auto-detection."""

    GARBAGE = {"none", "null", "undefined", "n/a", "okay.", "ok.", "yes.", "no."}

    def __init__(self, voice: str = "af_bella", device: str = "auto"):
        self.voice = voice
        self.device = device
        self._pipe = None
        self._available = False
        self._speak_lock = threading.Lock()
        self._sr = 24000

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

            self._pipe = KPipeline(lang_code='a')
            self._available = True
            self._gpu = (use_device == "cuda")
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
            return True
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
        if self._should_skip(text):
            return
        threading.Thread(target=self.speak, args=(text, True), daemon=True).start()

    @property
    def available(self) -> bool:
        return self._available


class WhisperSTT:
    """Speech-to-text via faster-whisper with hallucination filtering."""

    _HALLUCINATION_PHRASES = {
        "thank you", "thank you very much", "thank you so much",
        "thanks", "thanks for watching", "thank you for watching",
        "thank you for listening", "thanks for listening",
        "thank you for having me", "thank you for having us",
        "thanks for having me",
        "please subscribe", "like and subscribe", "don't forget to subscribe",
        "hit the like button", "leave a comment",
        "you", "the", ".", "", " ", "i", "a",
        "bye", "goodbye", "see you", "see you next time", "see you later",
        "uh", "um", "uh huh", "mm", "hmm", "ah", "oh", "yeah",
        "okay", "ok", "alright", "right", "sure",
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
            file_size = os.path.getsize(audio_path)
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
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=1.9,
                temperature=0.0,
            )

            good_segments = []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if hasattr(seg, "no_speech_prob") and seg.no_speech_prob > 0.55:
                    continue
                if self._is_hallucination(text):
                    continue
                good_segments.append(text)

            result = " ".join(good_segments).strip()

            if self._is_hallucination(result):
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


def print_banner():
    gpu = "GPU" if CONFIG["llm_n_gpu_layers"] > 0 else "CPU"
    print("\n\033[96m" + "═" * 52)
    print("   ╔═╗╔╗╔╔═╗╔╗╔╔╦╗╦ ╦╔╦╗")
    print("   ╠═╣║║║╠═╣║║║ ║ ║ ║║║║")
    print("   ╩ ╩╝╚╝╩ ╩╝╚╝ ╩ ╚═╝╩ ╩")
    print(f"   Edge AI Voice Assistant  [{gpu} Mode]")
    print("═" * 52 + "\033[0m\n")


class Anantum:
    """Main assistant orchestrator — loads models, runs voice/text loop."""

    def __init__(self):
        print_banner()

        # 1) TTS first so startup can announce progress.
        self.tts = KokoroTTS(voice=CONFIG["kokoro_voice"], device=CONFIG.get("tts_device", "auto"))
        tts_ok = self.tts.load()
        if tts_ok:
            self.tts.speak("Anantum starting up. Please wait.")
        else:
            print("[TTS] Text-only mode (install kokoro for voice)")

        # 2) Memory layer.
        from memory_system import MemoryManager
        self.memory = MemoryManager()

        # 3) STT layer.
        self.stt = WhisperSTT(CONFIG["whisper_model"],
                               CONFIG["stt_device"],
                               CONFIG["stt_compute"])
        stt_ok = self.stt.load()
        if not stt_ok:
            print("[STT] Text input fallback active")

        # 4) LLM in background so tool-only paths are available immediately.
        from llm_manager import LLMManager
        self.llm = LLMManager(
            model_path=CONFIG["llm_model"],
            n_ctx=CONFIG["llm_n_ctx"],
            n_threads=CONFIG["llm_n_threads"],
            n_gpu_layers=CONFIG["llm_n_gpu_layers"],
        )
        self._llm_ready = threading.Event()
        threading.Thread(target=self._load_llm_bg, daemon=True).start()

        # 5) Routing/response brain.
        from agent_brain import AgentBrain
        self.brain = AgentBrain(self.llm, self.memory)

        signal.signal(signal.SIGINT, self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

        gpu_str = f"GPU ({CONFIG['llm_n_gpu_layers']} layers)" if CONFIG["llm_n_gpu_layers"] > 0 else "CPU"
        print(f"\n[*] Anantum is online  [{gpu_str}]  — LLM warming up...\n")
        self.tts.speak("Ready. Instant tools available now. Language model loading in background.")

    def _load_llm_bg(self):
        try:
            self.llm.load()
            self._llm_ready.set()
            layers = CONFIG["llm_n_gpu_layers"]
            status = f"GPU ({layers} layers)" if layers > 0 else "CPU"
            print(f"\n[LLM] Gemma 3 1B ready on {status}")
            self.tts.speak("Language model ready. I'm fully operational now.")
        except FileNotFoundError as e:
            print(f"\n[LLM] Model not found: {e}")
            print("[LLM] Running tool-only mode.")
        except Exception as e:
            print(f"\n[LLM] Failed to load: {e}")

    def run_voice(self):
        print("[voice mode]  Ctrl+C to exit\n")

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
                return
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
        print("[text mode]  type 'exit' to quit\n")

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
        print("\n[Shutdown] Saving session...")
        self.memory.on_session_end()
        time.sleep(0.8)
        print("[Shutdown] Goodbye!")
        sys.exit(0)


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