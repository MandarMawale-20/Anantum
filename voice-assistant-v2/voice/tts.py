# Two-stage TTS pipeline: synth thread + playback thread.

import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from config.settings import CONFIG
from voice.base import BaseTTS

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTS):

    SKIP_PHRASES = {"none", "null", "undefined", "n/a", "okay.", "ok.", "yes.", "no."}

    # Common short responses for startup cache.
    CACHE_PHRASES = [
        "Sure.", "Done.", "Got it.", "One moment.",
        "Here you go.", "I'm not sure.", "Goodbye!", "All set.",
    ]

    def __init__(self, voice: str = "af_bella", device: str = "auto"):
        self.voice = voice
        self.device = device
        self._pipe = None
        self._available = False
        self._sr = 24000

        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue()
        self._cache: dict[str, np.ndarray] = {}
        self._gpu = False
        self._use_fp16 = False
        self._device = torch.device("cpu")

    def load(self) -> bool:
        try:
            from kokoro import KPipeline

            if self.device == "auto":
                use_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                use_device = self.device

            if use_device == "cuda":
                logger.info("Kokoro loading on GPU (CUDA)")
            else:
                logger.info("Kokoro loading on CPU")

            self._gpu = (use_device == "cuda")
            self._device = torch.device("cuda" if self._gpu else "cpu")
            self._use_fp16 = False

            self._pipe = KPipeline(lang_code='a')
            self._available = True

            # Keep FP32 for stability; some Kokoro internals create float32 inputs.
            if self._gpu:
                try:
                    if hasattr(self._pipe, "model") and hasattr(self._pipe.model, "to"):
                        self._pipe.model = self._pipe.model.to(self._device)
                    logger.info("Kokoro model moved to CUDA")
                except Exception as e:
                    logger.warning("CUDA model move failed, continuing with default placement: %s", e)

            # Start worker threads.
            threading.Thread(target=self._synth_worker, daemon=True).start()
            threading.Thread(target=self._play_worker, daemon=True).start()

            # Warm phrase cache in background.
            threading.Thread(target=self._warm_cache, daemon=True).start()

            logger.info("Kokoro ready (%s)", 'GPU' if self._gpu else 'CPU')
            return True

        except ImportError:
            logger.warning("kokoro not installed — pip install kokoro sounddevice")
            return False
        except Exception as e:
            logger.error("Failed to load TTS: %s", e)
            return False

    def _synth_worker(self):
        """Pull text from synth_q, synthesize, push audio array to play_q."""
        while True:
            text = self._synth_q.get()
            try:
                cached = self._cache.get(text.lower())
                if cached is not None:
                    self._play_q.put(cached)
                else:
                    audio = self._synthesize(text)
                    if audio is not None:
                        self._play_q.put(audio)
                    else:
                        logger.debug("Synthesis returned None for: %s", text)
            except Exception as e:
                logger.error("TTS synth worker error: %s", e, exc_info=True)
            finally:
                self._synth_q.task_done()

    def _play_worker(self):
        """Pull numpy arrays from play_q, play sequentially."""
        while True:
            audio = self._play_q.get()
            try:
                sd.play(audio, samplerate=self._sr)
                sd.wait()
            except Exception as e:
                logger.error("TTS play error: %s", e)
            finally:
                self._play_q.task_done()

    def _synthesize(self, text: str) -> np.ndarray | None:
        """Run kokoro pipeline, return concatenated numpy audio or None."""
        chunks = []
        try:
            with torch.no_grad():
                # Kokoro yields generator (graphemes, phonemes, audio)
                for _, _, audio in self._pipe(text, voice=self.voice):
                    if audio is not None and len(audio) > 0:
                        # Convert to numpy and ensure float32 for sounddevice playback
                        if hasattr(audio, "numpy"):
                            audio_np = audio.detach().cpu().numpy()
                        else:
                            audio_np = np.asarray(audio)
                            
                        if audio_np.dtype != np.float32:
                            audio_np = audio_np.astype(np.float32)
                            
                        chunks.append(audio_np)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "dtype" in error_msg or "not the same" in error_msg:
                logger.error("TTS dtype mismatch: %s. Attempting FP32 fallback.", e)
                try:
                    if hasattr(self._pipe, "model") and hasattr(self._pipe.model, "float"):
                        self._pipe.model = self._pipe.model.float()
                    self._use_fp16 = False
                    with torch.no_grad():
                        for _, _, audio in self._pipe(text, voice=self.voice):
                            if audio is not None and len(audio) > 0:
                                if hasattr(audio, "numpy"):
                                    audio_np = audio.detach().cpu().numpy()
                                else:
                                    audio_np = np.asarray(audio)
                                    
                                if audio_np.dtype != np.float32:
                                    audio_np = audio_np.astype(np.float32)
                                    
                                chunks.append(audio_np)
                except Exception as e2:
                    logger.error("Fallback synthesis also failed: %s", e2)
            else:
                logger.error("Synthesis error: %s", e)
        except Exception as e:
            logger.error("Unexpected TTS error: %s", e)

        if chunks:
            return np.concatenate(chunks)
        return None

    def _warm_cache(self):
        """Pre-synthesize common phrases so first use is instant."""
        time.sleep(0.5)  # let model settle after load
        for phrase in self.CACHE_PHRASES:
            try:
                audio = self._synthesize(phrase)
                if audio is not None:
                    self._cache[phrase.lower()] = audio
            except Exception as e:
                logger.debug("Failed to warm cache for '%s': %s", phrase, e)

    def _should_skip(self, text: str) -> bool:
        # Skip short, empty, bracketed, or garbage responses.
        if not text or not text.strip():
            return True
        t = text.strip()
        if len(t) < CONFIG.min_response_length:
            return True
        if t.startswith("[") and t.endswith("]"):
            return True
        if t.lower().rstrip(".,!?") in self.SKIP_PHRASES:
            return True
        return False

    def speak(self, text: str, blocking: bool = False):
        if self._should_skip(text):
            return
        text = text.strip()

        if not self._available or not self._pipe:
            logger.info("Anantum: %s", text)
            return

        self._synth_q.put(text)

        if blocking:
            self.wait_until_done()

    def wait_until_done(self):
        self._synth_q.join()
        self._play_q.join()

    def stop(self):
        try:
            sd.stop()
        except Exception:
            pass
        with self._synth_q.mutex:
            self._synth_q.queue.clear()
            self._synth_q.all_tasks_done.notify_all()
            self._synth_q.unfinished_tasks = 0
        with self._play_q.mutex:
            self._play_q.queue.clear()
            self._play_q.all_tasks_done.notify_all()
            self._play_q.unfinished_tasks = 0

    @property
    def available(self) -> bool:
        return self._available
