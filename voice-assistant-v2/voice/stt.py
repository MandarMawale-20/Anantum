# Whisper STT with hallucination filtering.

import logging
import os
import tempfile
import wave
from collections import Counter
from typing import Optional, Callable

from config.settings import CONFIG
from voice.base import BaseSTT

logger = logging.getLogger(__name__)


class WhisperSTT(BaseSTT):
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
            logger.info("Loading STT model %s on device=%s compute=%s...", self._model_name, self._device, self._compute)
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute,
            )
            logger.info("faster_whisper active device: %s", self._device)
            return True
        except ImportError:
            logger.warning("faster-whisper not installed: pip install faster-whisper")
            return False
        except Exception as e:
            logger.warning("Failed to load %s: %s", self._model_name, e)
            logger.info("Falling back to tiny.en...")
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel("tiny.en", device=self._device, compute_type=self._compute)
                logger.info("Loaded tiny.en as fallback")
                return True
            except Exception as e2:
                logger.error("Fallback also failed: %s", e2)
                return False

    def _is_hallucination(self, text: str) -> bool:
        if not text:
            return True
        t = text.lower().strip().rstrip(".,!?")
        if t in self._HALLUCINATION_PHRASES:
            return True
        # Catch repetitive junk like "thank you thank you".
        words = t.split()
        if len(words) >= 4:
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if t.count(phrase) >= 3:
                    return True
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
                    logger.debug("Rejected (no_speech=%.2f): %r", seg.no_speech_prob, text)
                    continue
                if self._is_hallucination(text):
                    logger.debug("Rejected (hallucination): %r", text)
                    continue
                good_segments.append(text)

            result = " ".join(good_segments).strip()

            if self._is_hallucination(result):
                logger.debug("Rejected (final filter): %r", result)
                return ""

            return result
        except Exception as e:
            logger.error("STT error: %s", e)
            return ""

    def record(self, on_speech_start: Optional[Callable[[], None]] = None) -> Optional[str]:
        try:
            import sounddevice as sd
            import numpy as np

            sr = CONFIG.sample_rate
            chunk_size = int(sr * 0.1)
            threshold = CONFIG.silence_threshold
            max_silent = int(CONFIG.silence_duration / 0.1)
            min_chunks = CONFIG.min_speech_chunks
            max_chunks = int(CONFIG.max_record_seconds / 0.1)

            logger.debug("Listening...")

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
                            logger.debug("Speech detected")
                            speech_started = True
                            if on_speech_start is not None:
                                try:
                                    on_speech_start()
                                except Exception:
                                    pass
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
            logger.error("Recording error: %s", e)
            return None
