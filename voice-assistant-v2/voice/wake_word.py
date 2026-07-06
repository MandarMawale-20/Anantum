import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


class OpenWakeWordListener:
    def __init__(self, model_paths: tuple[str, ...], threshold: float = 0.5):
        self.model_paths = tuple(p for p in model_paths if p)
        self.threshold = threshold

        self._model = None
        self._pa = None
        self._stream = None
        self._thread = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

    @property
    def available(self) -> bool:
        return self._model is not None and self._stream is not None

    def start(self) -> bool:
        try:
            from openwakeword.model import Model
            import pyaudio
        except Exception as e:
            logger.warning("Wake listener unavailable (missing dependency): %s", e)
            return False

        valid_paths = tuple(p for p in self.model_paths if os.path.exists(p))
        if not valid_paths:
            logger.warning("Wake listener disabled: no OpenWakeWord model found at configured paths")
            return False

        try:
            self._model = Model(
                wakeword_models=list(valid_paths),
                inference_framework="onnx",
            )
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                rate=16000,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=1280,
            )
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info("Wake listener started on CPU with OpenWakeWord (%d models)", len(valid_paths))
            return True
        except Exception as e:
            logger.warning("Wake listener init failed: %s", e)
            self.stop()
            return False

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pcm = self._stream.read(1280, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                scores = self._model.predict(pcm)
                if any(float(v) >= self.threshold for v in scores.values()):
                    self._wake_event.set()
            except OSError:
                break
            except Exception:
                continue

    def wait_for_wake(self, timeout: float = None) -> bool:
        ok = self._wake_event.wait(timeout=timeout)
        if ok:
            self._wake_event.clear()
        return ok

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        try:
            if self._model is not None:
                self._model = None
        except Exception:
            pass
        self._stream = None
        self._pa = None
        self._model = None
