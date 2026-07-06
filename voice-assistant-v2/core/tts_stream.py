import logging
import queue
import threading
import time


logger = logging.getLogger(__name__)


class StreamingTTS:
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def push_text(self, text: str) -> None:
        if text and text.strip():
            self.queue.put(text.strip())

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.tts.stop()
        except Exception:
            pass
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.all_tasks_done.notify_all()
            self.queue.unfinished_tasks = 0
        self._stop_event.clear()

    def run(self) -> None:
        while True:
            try:
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.tts.speak(text, blocking=False)
                deadline = time.time() + 30
                while self.tts._play_q.unfinished_tasks > 0:
                    if time.time() > deadline:
                        logger.warning("TTS playback timed out, skipping")
                        self.tts.stop()
                        break
                    time.sleep(0.05)
            except Exception as e:
                logger.error("StreamingTTS run error: %s", e)
            finally:
                self.queue.task_done()

    def wait_until_done(self) -> None:
        self.queue.join()
