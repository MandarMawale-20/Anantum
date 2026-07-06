# Runtime configuration.

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from config.user_settings import get_settings_file, load_user_settings, save_user_settings

try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    _dotenv_loaded = False


def _env(key: str, default: str = "") -> str:
    """Read an env var, returning default if unset or empty."""
    val = os.environ.get(key, "").strip()
    return val if val else default


# Configure app-wide logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    # LLM defaults
    llm_model: str = "models/gemma3-voice-Q5_K_M.gguf"
    llm_n_ctx: int = 2048
    llm_n_threads: int = 8
    llm_n_gpu_layers: int = 40
    
    # Voice/STT defaults
    kokoro_voice: str = "af_sky"
    tts_device: str = "cpu"
    whisper_model: str = "distil-small.en"
    stt_device: str = "cpu"
    stt_compute: str = "int8"
    
    # Memory
    warm_store_max_entries: int = 5000
    summary_threshold: int = 20
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Privacy
    allow_ip_geolocation: bool = False

    # Audio capture
    sample_rate: int = 24000
    silence_threshold: float = 500.0
    silence_duration: float = 0.8
    min_speech_chunks: int = 4
    max_record_seconds: float = 15.0

    # Skip short response fragments
    min_response_length: int = 1
    
    # TTS config
    tts_dropout: float = 0.0

    # Wake word
    wake_word_enabled: bool = True
    wake_word_terms: tuple[str, ...] = (
        "anantum",
        "anantam",
        "anandum",
        "anant",
    )
    wake_word_model_paths: tuple[str, ...] = (
        "models/wake/anantum.onnx",
        "models/wake/anantam.onnx",
        "models/wake/anandum.onnx",
    )
    wake_word_threshold: float = 0.5

    # Data directory
    data_dir: Path = None

    # Internal: tracks whether user explicitly set gpu_layers (avoids auto-detect override)
    _gpu_layers_explicitly_set: bool = False
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = Path("data")
            self.data_dir.mkdir(exist_ok=True)
        self._apply_env_overrides()
        self._validate()
        self.apply_persisted_user_settings()
        self._validate()  # re-validate after user settings are applied
        self._auto_detect_wake_word()

    def _apply_env_overrides(self) -> None:
        """Apply values from .env file so the user's file actually takes effect."""
        model_path = _env("LLM_MODEL_PATH")
        if model_path:
            self.llm_model = model_path
        gpu = _env("LLM_N_GPU_LAYERS")
        if gpu:
            try:
                self.llm_n_gpu_layers = max(0, int(gpu))
            except ValueError:
                pass
        ctx = _env("LLM_N_CTX")
        if ctx:
            try:
                self.llm_n_ctx = max(512, int(ctx))
            except ValueError:
                pass
        threads = _env("LLM_N_THREADS")
        if threads:
            try:
                self.llm_n_threads = max(1, int(threads))
            except ValueError:
                pass
        voice = _env("TTS_VOICE")
        if voice:
            self.kokoro_voice = voice
        tts_dev = _env("TTS_DEVICE")
        if tts_dev in ("auto", "cuda", "cpu"):
            self.tts_device = tts_dev
        stt_dev = _env("STT_DEVICE")
        if stt_dev in ("auto", "cuda", "cpu"):
            self.stt_device = stt_dev
        stt_model = _env("STT_MODEL")
        if stt_model:
            self.whisper_model = stt_model
        stt_comp = _env("STT_COMPUTE")
        if stt_comp:
            self.stt_compute = stt_comp
        wake_enabled = _env("WAKE_WORD_ENABLED")
        if wake_enabled:
            self.wake_word_enabled = wake_enabled.lower() in ("true", "1", "yes")
        wake_threshold = _env("WAKE_WORD_THRESHOLD")
        if wake_threshold:
            try:
                self.wake_word_threshold = max(0.0, min(1.0, float(wake_threshold)))
            except ValueError:
                pass
        wake_models = _env("WAKE_WORD_MODELS")
        if wake_models:
            parts = [p.strip() for p in wake_models.split(";") if p.strip()]
            if parts:
                self.wake_word_model_paths = tuple(parts)
        sr = _env("SAMPLE_RATE")
        if sr:
            try:
                self.sample_rate = int(sr)
            except ValueError:
                pass
        silence_db = _env("SILENCE_THRESHOLD")
        if silence_db:
            try:
                self.silence_threshold = float(silence_db)
            except ValueError:
                pass
        silence_dur = _env("SILENCE_DURATION")
        if silence_dur:
            try:
                self.silence_duration = max(0.1, float(silence_dur))
            except ValueError:
                pass
        min_chunks = _env("MIN_SPEECH_CHUNKS")
        if min_chunks:
            try:
                self.min_speech_chunks = max(1, int(min_chunks))
            except ValueError:
                pass
        max_rec = _env("MAX_RECORD_SECONDS")
        if max_rec:
            try:
                self.max_record_seconds = float(max_rec)
            except ValueError:
                pass
        min_resp = _env("MIN_RESPONSE_LENGTH")
        if min_resp:
            try:
                self.min_response_length = int(min_resp)
            except ValueError:
                pass

    def _auto_detect_wake_word(self) -> None:
        """Disable wake word if the model files don't exist on disk."""
        if not self.wake_word_enabled:
            return
        missing = [p for p in self.wake_word_model_paths if not Path(p).exists()]
        if missing:
            logger.warning(
                "Wake word model(s) not found: %s. Disabling wake word.",
                "; ".join(missing),
            )
            self.wake_word_enabled = False
            logger.info("Wake word disabled. Assistant will use push-to-talk / click-to-talk.")

    def _validate(self) -> None:
        corrected = []
        if self.llm_n_gpu_layers < 0:
            self.llm_n_gpu_layers = 0
            corrected.append("gpu_layers was negative, reset to 0")
        if self.tts_device not in ("auto", "cuda", "cpu"):
            original = self.tts_device
            self.tts_device = "auto"
            corrected.append(f"invalid tts_device '{original}', reset to auto")
        if self.llm_n_ctx < 512:
            original = self.llm_n_ctx
            self.llm_n_ctx = 512
            corrected.append(f"n_ctx too small ({original}), reset to 512")
        if self.silence_duration <= 0:
            original = self.silence_duration
            self.silence_duration = 0.8
            corrected.append(f"silence_duration was <= 0 ({original}), reset to 0.8")
        if self.min_speech_chunks < 1:
            original = self.min_speech_chunks
            self.min_speech_chunks = 4
            corrected.append(f"min_speech_chunks was < 1 ({original}), reset to 4")
        for msg in corrected:
            logger.warning("Config corrected: %s", msg)

    def apply_persisted_user_settings(self) -> None:
        payload = load_user_settings()
        if not payload:
            return

        model_path = str(payload.get("model_path", "")).strip()
        if model_path:
            self.llm_model = model_path

        voice = str(payload.get("voice", "")).strip()
        if voice:
            self.kokoro_voice = voice

        tts_device = str(payload.get("tts_device", "")).strip().lower()
        if tts_device in {"auto", "cuda", "cpu"}:
            self.tts_device = tts_device

        stt_device = str(payload.get("stt_device", "")).strip().lower()
        if stt_device in {"cuda", "cpu", "auto"}:
            self.stt_device = stt_device

        gpu_layers = payload.get("gpu_layers")
        if isinstance(gpu_layers, int):
            self.llm_n_gpu_layers = max(0, gpu_layers)
            self._gpu_layers_explicitly_set = True

    def to_user_settings_dict(self) -> dict:
        return {
            "model_path": self.llm_model,
            "voice": self.kokoro_voice,
            "gpu_layers": self.llm_n_gpu_layers,
            "tts_device": self.tts_device,
            "stt_device": self.stt_device,
        }

    def save_user_settings(self) -> tuple[bool, str | None]:
        return save_user_settings(self.to_user_settings_dict())

    @property
    def user_settings_file(self) -> Path:
        return get_settings_file()
    
    @property
    def NOTES_DB_FILE(self) -> Path:
        return self.data_dir / "notes.db"

    @property
    def COLD_ARCHIVE_DB(self) -> Path:
        return self.data_dir / "cold_archive.db"

    @property
    def WARM_FAISS_INDEX(self) -> Path:
        return self.data_dir / "warm_index.faiss"

    @property
    def WARM_METADATA_FILE(self) -> Path:
        return self.data_dir / "warm_meta.json"
    
    def save_user_overrides(self) -> bool:
        ok, _ = self.save_user_settings()
        return ok


CONFIG = AppConfig()
