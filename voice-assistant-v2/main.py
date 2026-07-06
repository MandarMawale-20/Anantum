import argparse
import logging
import sys
import threading

from logging.handlers import RotatingFileHandler

try:
    from colorama import init
    init(autoreset=True)
except ImportError:
    pass


def _configure_root_logging() -> None:
    """Send runtime logs to a rotating file in the app data directory."""
    from config.settings import CONFIG

    log_file = CONFIG.data_dir / "anantum.log"
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def _log_thread_exception(args):
    """Surface background-thread crashes in the main log."""
    logging.critical(
        "Unhandled exception in thread '%s': %s",
        args.thread.name if args.thread else "unknown",
        args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anantum - local offline voice assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --mode text
  python main.py --gpu 0
  python main.py --gpu 20 --voice af
  python main.py --model models/custom.gguf
  python main.py --tts-device cuda
  python main.py --wake-model models/wake/anantum.onnx --wake-model models/wake/anantam.onnx
        """
    )

    parser.add_argument(
        "--mode",
        choices=["voice", "text", "bridge"],
        default="voice",
        help="Assistant mode to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the GGUF model path"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU layers to offload; 0 forces CPU"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Kokoro voice code"
    )
    parser.add_argument(
        "--tts-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="TTS device"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Alias for --mode text"
    )
    parser.add_argument(
        "--wake-word",
        action="store_true",
        help="Enable wake-word gating"
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Disable wake-word gating"
    )
    parser.add_argument(
        "--wake-model",
        action="append",
        default=None,
        help="OpenWakeWord ONNX model path; repeat for multiple models"
    )
    return parser


def _apply_cli_overrides(args) -> None:
    from config.settings import CONFIG

    changed = False

    if args.model:
        CONFIG.llm_model = args.model
        changed = True
    if args.gpu is not None:
        CONFIG.llm_n_gpu_layers = args.gpu
        CONFIG._gpu_layers_explicit = True
        changed = True
    if args.voice:
        CONFIG.kokoro_voice = args.voice
        changed = True
    if args.tts_device != "auto":
        CONFIG.tts_device = args.tts_device
        changed = True
    if args.wake_word:
        CONFIG.wake_word_enabled = True
        changed = True
    if args.no_wake_word:
        CONFIG.wake_word_enabled = False
        changed = True
    if args.wake_model:
        CONFIG.wake_word_model_paths = tuple(args.wake_model)
        changed = True

    if changed:
        CONFIG.save_user_overrides()


def main():
    threading.excepthook = _log_thread_exception
    _configure_root_logging()

    parser = _build_parser()
    args = parser.parse_args()

    if args.text:
        args.mode = "text"

    _apply_cli_overrides(args)

    try:
        if args.mode == "bridge":
            from bridge.stdio_bridge import run_stdio_bridge

            run_stdio_bridge()
        else:
            from core.assistant import Anantum

            assistant = Anantum()

            if args.mode == "text":
                assistant.run_text()
            else:
                assistant.run_voice()

    except KeyboardInterrupt:
        print("\n\nShutdown requested.")
        sys.exit(0)
    except Exception as exc:
        logging.exception("Fatal error during startup or execution: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
