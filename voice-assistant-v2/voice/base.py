# Abstract interfaces for voice I/O engines.

from abc import ABC, abstractmethod
from typing import Optional


class BaseTTS(ABC):
    """Abstract text-to-speech engine. Concrete classes implement synthesis and playback."""

    @abstractmethod
    def load(self) -> bool:
        ...

    @abstractmethod
    def speak(self, text: str, blocking: bool = False) -> None:
        ...

    @abstractmethod
    def wait_until_done(self) -> None:
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        ...


class BaseSTT(ABC):
    """Abstract speech-to-text engine. Concrete classes implement the mic and transcription."""

    @abstractmethod
    def load(self) -> bool:
        ...

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        ...

    @abstractmethod
    def record(self) -> Optional[str]:
        ...
