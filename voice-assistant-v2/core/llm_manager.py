# Llama.cpp wrapper for prompt building and generation.

import logging
import re
from typing import Optional, Callable
from pathlib import Path

from config.settings import CONFIG

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)


def sanitize_response(text: str) -> str:
    """Remove leaked control/meta tokens from model output."""
    if not text:
        return ""
    cleaned = text
    cleaned = cleaned.replace("<end_of_turn>", "")
    cleaned = re.sub(r"<start_of_turn>[^\n]*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*intent\s*:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*tool\s*:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def format_turn(role: str, content: str) -> str:
    return f"<start_of_turn>{role}\n{content}<end_of_turn>\n"


def format_generation_start(role: str = "model") -> str:
    return f"<start_of_turn>{role}\n"


PAST_SUMMARY_BLOCK = """[Previous conversation summary]
{summaries}

"""


class LLMManager:
    """Model lifecycle and generation entrypoint."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8, n_gpu_layers: int = 0):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        logger.info("LLMManager initialized (model: %s)", self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    def load(self) -> bool:
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            return False
        
        if Llama is None:
            logger.error("llama-cpp-python not installed")
            return False
        
        try:
            logger.info("Loading Gemma 3.1 1B from %s", self.model_path)
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=1024,
                verbose=False,
                chat_format="gemma"
            )
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def build_prompt(
        self,
        user_message: str,
        chat_history: list = None,
        past_summaries: str = None
    ) -> str:
        if chat_history is None:
            chat_history = []
        
        prompt = ""
        
        if past_summaries:
            prompt += PAST_SUMMARY_BLOCK.format(summaries=past_summaries)
        
        for turn in chat_history:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
            else:
                role = "user"
                content = str(turn)
            prompt += format_turn(role, content)
        
        prompt += format_turn("user", user_message)
        prompt += format_generation_start("model")
        
        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        if self.llm is None:
            logger.error("Model not loaded")
            return ""

        try:
            response = ""
            kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["<end_of_turn>"],
            }

            if on_token:
                for token in self.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature):
                    if token:
                        response += token
                        should_continue = on_token(token)
                        if should_continue is False:
                            break
                return sanitize_response(response)

            output = self.llm(prompt, stream=False, **kwargs)
            text = output.get("choices", [{}])[0].get("text", "") if isinstance(output, dict) else str(output)
            return sanitize_response(text)
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return ""

    def stream_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        if self.llm is None:
            return
        kwargs = {
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["<end_of_turn>"],
        }
        for output in self.llm(prompt, **kwargs):
            token = output.get("choices", [{}])[0].get("text", "")
            if token:
                yield token

    def unload(self) -> None:
        self.llm = None
        logger.info("Model unloaded")
