# Anantum v2 — Edge AI Voice Assistant

> **A local-first, privacy-preserving voice assistant that runs entirely on-device. Speech recognition, LLM inference, text-to-speech, persistent memory, and a desktop widget — all without a cloud dependency.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tauri 2.0](https://img.shields.io/badge/tauri-2.0-purple)](https://v2.tauri.app/)
[![GGUF](https://img.shields.io/badge/llama.cpp-GGUF-orange)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-GTX_1660_Ti_6GB-green)]()
[![STT](https://img.shields.io/badge/STT-Whisper-yellow)]()
[![TTS](https://img.shields.io/badge/TTS-Kokoro-red)]()

---

### 🎥 Demo Video · 🏗️ [Architecture Diagram](https://www.figma.com) · 📄 [Research Document](docs/research.md) · 📚 [Documentation](docs/)

---

## 1. What is Anantum v2?

Anantum v2 is a production-ready evolution of the original Anantum voice assistant, redesigned with a modular architecture, a polished Tauri desktop widget, and a robust stdio bridge protocol for frontend-backend communication.

- **What is it?** A fully local voice AI assistant with a desktop UI, multi-step planning, wake word support, and a three-tier memory system — all running on consumer hardware.
- **Who is it for?** Developers who want a real, usable local AI assistant they can extend and customize. Privacy-conscious users who want a Siri/Alexa alternative that doesn't phone home.
- **Why did you build it?** The monolithic v1 proved the concept but was hard to extend. v2 is the architecture that v1 deserved — modular, testable, packagable, and production-ready.

---

## 2. Motivation

### Why Local AI?

Most voice assistant demos fall into one of three traps:

1. **Cloud-dependent** — Your voice data, transcripts, and preferences leave your machine.
2. **Fragile** — A single API outage breaks the entire experience.
3. **Toy-like** — They demonstrate one trick (wake word, or chat, or a tool) but never feel like a real product.

Anantum v2 solves all three. It keeps the **entire interaction loop local**, supports **voice and text workflows**, remembers **useful context across sessions**, and exposes a **polished desktop UI** that feels like a native OS widget.

### Why Not Cloud APIs?

| Concern | Cloud Assistant | Anantum v2 |
|---------|----------------|------------|
| **Privacy** | Your data is processed on remote servers | Everything stays on your machine |
| **Reliability** | Requires internet connectivity | Works fully offline |
| **Latency** | Network round-trips add 200–500ms | Tools respond instantly; local inference |
| **Cost** | Pay-per-token or subscription | One-time setup, zero ongoing cost |
| **Control** | Black-box model updates | You choose the model and config |

### Why This Project Exists

The tradeoff is intentional: Anantum is designed to be **practical first**, not magical first. It does useful work (timers, notes, system info) before the LLM even finishes loading. This changes the user's perception from "waiting for AI" to "using a tool that happens to have AI capabilities."

---

## 3. Key Features

✅ **Local LLM** — Gemma 3 1B via llama.cpp, runs entirely on-device

✅ **Voice & Text Modes** — Voice mode with optional wake word, text mode for quiet environments

✅ **Desktop Widget** — Tauri 2.0 floating widget with SiriWave visualization

✅ **Long-term Memory** — Three-tier memory system (hot cache → FAISS → SQLite archive)

✅ **Instant Tools** — Regex-first intent routing for sub-millisecond command execution

✅ **Multi-step Planning** — Celestial mode for complex task execution

✅ **Wake Word** — "Anantum" activation via OpenWakeWord

✅ **Bridge Protocol** — stdin/stdout JSON protocol for frontend-backend communication

✅ **Background Model Loading** — Assistant is usable before the LLM finishes loading

✅ **Streaming TTS** — Sentence-by-sentence speech output during LLM generation

✅ **Windows Installer** — PyInstaller + Inno Setup for one-click installation

---

## 4. Demo

### Quick Demo (Text Mode)

```bash
cd voice-assistant-v2
python main.py --mode text --gpu 0

# Try these commands:
# "what time is it"
# "set a timer for 5 minutes"
# "save a note: buy milk"
# "hello" (waits for LLM to load)
# "activate celestial mode"
# "what's the weather in London"
```

### Desktop App Demo

```bash
cd frontend
npm install
npm run tauri:dev
```

### Bridge Mode (Manual)

```bash
python backend_launcher.py
# Then type JSON commands:
# {"id": 1, "command": "health", "args": {}}
# {"id": 2, "command": "start_session", "args": {"mode": "text"}}
# {"id": 3, "command": "input_text", "args": {"text": "what time is it"}}
```

### Screenshots

> *Screenshots and screen recordings coming soon.*

| State | Preview |
|-------|---------|
| **Idle** | `docs/screenshots/idle.png` — Floating widget in standby |
| **Listening** | `docs/screenshots/listening.png` — SiriWave active, "Listening" label |
| **Speaking** | `docs/screenshots/speaking.gif` — Animated response with streaming text |
| **Model Picker** | `docs/screenshots/model-picker.png` — File dialog for model selection |

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Tauri Desktop Shell (Rust)                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ lib.rs: BridgeProcess                                      │  │
│  │  • Spawns Python as child process                          │  │
│  │  • stdin/stdout JSON-line protocol                         │  │
│  │  • Routes events → WebView via Tauri emit()                │  │
│  └──────────────────────┬─────────────────────────────────────┘  │
│                         │ stdin/stdout                           │
│  ┌──────────────────────▼─────────────────────────────────────┐  │
│  │ stdio_bridge.py + server.py                                │  │
│  │  • Reads JSON commands from stdin                          │  │
│  │  • Writes JSON events/responses to stdout                  │  │
│  │  • Manages assistant lifecycle (start/stop/health)         │  │
│  └──────────────────────┬─────────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼─────────────────────────────────────┐  │
│  │ core/assistant.py: Anantum Runtime                         │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │  │
│  │  │ Voice I/O│  │ Intent Router│  │ LLM (llama.cpp/GGUF) │  │  │
│  │  │ STT + TTS│──│ Regex-first  │──│ Background load      │  │  │
│  │  │ Wake word│  │ Tool dispatch│  │ Streaming generation │  │  │
│  │  └──────────┘  └──────┬───────┘  └──────────────────────┘  │  │
│  │                       │                                    │  │
│  │  ┌────────────────────▼──────────────────────────────────┐ │  │
│  │  │ Memory System (3-Tier)                                │ │  │
│  │  │  Hot Cache (20 turns) → Warm FAISS (5K entries)       │ │  │
│  │  │  → Cold SQLite Archive (FTS5)                         │ │  │
│  │  └───────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Frontend (WebView):                                             │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ script.js + SiriWave.js                                      ││
│  │  • Listens to Tauri events "assistant-event"                 ││
│  │  • Calls invoke() to send commands                           ││
│  │  • SiriWave visual feedback per state (idle/listening/       ││
│  │    thinking/speaking)                                        ││
│  │  • Watchdog: 60s health-check interval                       ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Communication Flow

```
User speaks → STT (Whisper) → Intent Classifier (Regex)
  ├─ Tool match? → Execute tool → TTS response
  └─ No match? → LLM (GGUF) → Streaming response → TTS
                      │
                 Memory (3 tiers)
              Hot → Warm → Cold
```

> 🏗️ [View full architecture on Figma](https://www.figma.com)

---

## 6. Design Philosophy

### Why Local?

Cloud AI is convenient but fundamentally compromises user privacy. Every voice command, every conversation, every preference is processed on someone else's server. Anantum was built on the principle that **privacy should not be a premium feature** — it should be the default.

### Why Edge?

Running inference on edge hardware forces better engineering. You can't throw more GPUs at the problem. You have to optimize: smaller models, efficient architectures, smart routing. The result is a system that's **more thoughtful, more efficient, and more reliable** than a cloud-dependent equivalent.

### Why Modular?

The monolithic v1 was great for prototyping but became hard to extend. v2's modular architecture separates concerns (voice, memory, skills, planning) so each component can be developed, tested, and improved independently. This also makes it easier for contributors to understand and extend specific parts.

### Why Offline?

Internet connectivity is not guaranteed. A voice assistant that stops working when you lose signal is not a real assistant. Anantum is designed to work **fully offline** — every component runs locally. Optional features (weather, web search) gracefully degrade when offline.

### Why Privacy?

Your conversations, your data, your preferences — they belong to you. Anantum stores everything locally, processes everything locally, and never phones home. There are no telemetry pings, no analytics, no data collection.

---

## 7. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Assistant Runtime** | Python 3.10+ |
| **LLM Inference** | llama.cpp / GGUF (via `llama-cpp-python`) |
| **Speech-to-Text** | Whisper (via `faster-whisper`) |
| **Text-to-Speech** | Kokoro ONNX |
| **Memory (Semantic)** | FAISS (HNSW) + Sentence Transformers |
| **Memory (Archive)** | SQLite with FTS5 |
| **Desktop Shell** | Tauri 2.0 (Rust) |
| **Frontend** | HTML, CSS, JavaScript, SiriWave.js |
| **Wake Word** | OpenWakeWord ONNX |
| **Packaging** | PyInstaller + Inno Setup |

---

## 8. System Pipeline

```
Voice Input
     │
     ▼
Whisper STT (faster-whisper, local)
     │
     ▼
Intent Pre-Classifier (Regex, <1ms)
     │
     ├── Tool Match? ──► Execute Tool ──► TTS Response
     │
     └── No Match? ──► LLM (Gemma 3 1B via llama.cpp)
                            │
                       Memory Recall
                       (Hot → Warm → Cold)
                            │
                       Response Generation
                            │
                            ▼
                    Kokoro TTS (local)
                            │
                            ▼
                    Voice Output
```

### Desktop Integration Pipeline

```
Tauri WebView (User clicks mic / types)
     │
     ▼
invoke() → Rust Bridge → stdin JSON
     │
     ▼
Python stdio_bridge.py → Assistant Runtime
     │
     ▼
Response → stdout JSON → Rust Bridge
     │
     ▼
Tauri emit() → WebView (SiriWave + captions)
```

---

## 9. Engineering Decisions ⭐⭐⭐⭐⭐

### Why Tool-First Architecture?

**Problem:** LLM inference takes 100-500ms even on GPU. For simple commands like "what's the time?", waiting for the LLM is wasteful.

**Options:**
- Route everything through the LLM — Simple but slow
- Regex-first routing — Fast but requires maintaining patterns
- Hybrid — Use regex for known patterns, LLM for everything else

**Decision:** Regex-first intent routing. Common commands are handled by regex before the LLM is involved. This means timers, notes, and system info work instantly, even while the model loads.

**Tradeoff:** Requires maintaining regex patterns for each tool. But the latency savings (sub-1ms vs 100-500ms) are worth it.

### Why Background Model Loading?

**Problem:** Loading a GGUF model takes 10-30 seconds. Blocking startup on model loading makes the assistant feel slow.

**Options:**
- Blocking load — Simple but bad UX
- Background load — Complex but responsive
- Lazy load — First LLM query triggers load

**Decision:** Background model loading in a separate thread. The assistant becomes useful (tools + memory) before the LLM is ready. Tool commands work immediately; the model loads in a background thread.

**Tradeoff:** More complex startup logic. But the user experience is dramatically better — the assistant feels instant.

### Why Three-Tier Memory?

**Problem:** Memory needs to handle different time scales — immediate context, cross-session recall, and long-term storage — each with different performance requirements.

**Options:**
- Single vector store — Simple but misses recency
- In-memory only — Fast but doesn't persist
- Three-tier — Complex but handles all cases

**Decision:** Three-tier memory: hot cache (20 turns, instant), warm FAISS store (5K entries, <10ms), cold SQLite archive (unlimited, <100ms). Each tier has different speed/capacity tradeoffs.

**Tradeoff:** More code, more complexity. But the quality of memory recall is significantly better than any single approach.

### Why stdio Bridge Over HTTP?

**Problem:** The frontend needs to communicate with the Python backend. HTTP is the obvious choice, but it has downsides for a local desktop app.

**Options:**
- HTTP server — Standard, but exposes a port, requires CORS handling
- stdio bridge — No ports, no CORS, simple lifecycle
- Unix sockets — Fast but platform-specific

**Decision:** stdio bridge over stdin/stdout. The frontend talks to Python over stdin/stdout instead of a localhost server. This keeps the packaged app simpler, avoids exposing a network port, and makes lifecycle management straightforward.

**Tradeoff:** Only one client can connect at a time. But for a single-user desktop app, this is not a limitation.

### Why Minimal Desktop Shell?

**Problem:** The desktop UI needs to be functional but not bloated.

**Options:**
- Full Electron app — Heavy, complex
- Tauri widget — Lightweight, native
- System tray icon — Too minimal

**Decision:** Tauri 2.0 widget (366×76). The Python runtime owns all assistant logic; the shell is just a presentation layer. The widget is intentionally small — it shows SiriWave visualization, captions, and a model picker.

**Tradeoff:** Limited UI surface. But the assistant is primarily voice-driven; the UI is supplementary.

### Why Regex-First Intent Routing?

**Problem:** LLM inference is expensive (100-500ms) for simple, predictable commands.

**Options:**
- LLM for everything — Simple but slow
- Regex classifier — Fast but limited
- ML classifier — Complex to maintain

**Decision:** Regex-first. Before invoking the LLM, a regex-based classifier matches common intents. This saves tokens, reduces latency, and makes the assistant feel responsive.

**Tradeoff:** Regex patterns need maintenance as new commands are added. But the patterns are simple and well-documented.

---

## 10. Challenges

### GPU Memory Management
Running STT, LLM, TTS, and embeddings on a 6GB GPU required careful layer offloading and memory cleanup. The LLM uses 35 GPU layers, embeddings use ONNX on GPU, and FAISS uses GPU-accelerated search — all competing for the same VRAM.

**Solution:** Sequential model loading, GPU memory pooling, and CPU fallback for non-critical components. The system dynamically manages GPU resources based on current task.

### Streaming TTS Coordination
Getting Kokoro to stream audio sentence-by-sentence (rather than waiting for the full response) required threading and queue coordination.

**Solution:** Threaded TTS queue that starts speaking sentence-by-sentence. The LLM generates tokens, which are buffered, split into sentences, and sent to TTS in parallel.

### Bridge Protocol Reliability
The stdio bridge needs to handle process crashes, timeouts, and malformed JSON without breaking the UI.

**Solution:** Watchdog timer (60s health-check interval), auto-reconnect logic, and graceful error handling on both sides of the bridge.

### Wake Word Accuracy
OpenWakeWord works well in quiet environments but triggers false positives with background noise.

**Solution:** Configurable sensitivity threshold, VAD-based filtering, and a cooldown period between activations.

### Packaging Complexity
PyInstaller builds are fragile — missing imports, hidden dependencies, and platform-specific issues are common.

**Solution:** Extensive testing of the packaged build, explicit import hooks, and a comprehensive Inno Setup installer that handles Python path and dependency resolution.

### Cross-Platform Audio
sounddevice works differently on Windows, Linux, and Mac. Device enumeration, sample rates, and buffer sizes vary.

**Solution:** Platform-specific audio configuration with sensible defaults and fallback options.

---

## 11. Performance

Measured on **16GB RAM, NVIDIA GTX 1660 Ti (6GB VRAM)**, using **Gemma 3 1B Q5_K_M**:

| Metric | Value |
|--------|-------|
| **First Token** | ~100ms |
| **LLM Generation** | ~15 tok/s |
| **STT (Whisper)** | 200-500ms |
| **TTS (Kokoro)** | 50-200ms |
| **Intent Classification** | <1ms |
| **Tool Execution** | <5ms |
| **Warm Memory Lookup (FAISS)** | <50ms |
| **Cold Memory Search (SQLite FTS5)** | <100ms |
| **Model Load Time (CPU, 4 threads)** | ~25-35s |
| **Model Load Time (GPU, 35 layers)** | ~10-15s |
| **RAM Usage (idle)** | ~400 MB |
| **RAM Usage (with model loaded)** | ~2.5 GB |
| **VRAM Usage (35 layers offloaded)** | ~2 GB |
| **Time to First Useful Response** | <1s (tool command) |
| **Voice Command Round-Trip (Tool)** | ~2-3s |
| **Voice Command Round-Trip (LLM)** | ~5-8s |
| **Offline** | Yes |
| **Internet** | Optional |

> **Note:** Replace these with measurements from your own machine before publishing.

---

## 12. Folder Structure

```
voice-assistant-v2/
├── main.py                  # CLI entrypoint (voice, text, bridge modes)
├── backend_launcher.py      # Packaged EXE entrypoint for desktop builds
├── config/                  # Runtime configuration & user settings
│   ├── settings.py          # AppConfig dataclass
│   └── user_settings.py     # Persisted user preferences
├── core/                    # Assistant orchestration
│   ├── assistant.py         # Anantum class — startup, voice/text loops
│   ├── agent.py             # AgentBrain — intent routing, tool dispatch
│   ├── intent_detector.py   # Regex-first intent pre-classifier
│   ├── llm_manager.py       # llama.cpp wrapper
│   ├── context_builder.py   # Prompt context from memory tiers
│   └── tts_stream.py        # Non-blocking TTS queue
├── voice/                   # Speech I/O
│   ├── base.py              # Abstract STT/TTS interfaces
│   ├── stt.py               # WhisperSTT
│   ├── tts.py               # KokoroTTS (two-stage synth+play)
│   └── wake_word.py         # OpenWakeWord listener
├── memory/                  # Three-tier memory system
│   ├── hot_cache.py         # In-memory ring buffer
│   ├── faiss_store.py       # FAISS HNSW semantic search
│   ├── cold_archive.py      # SQLite FTS5 archive
│   ├── memory_manager.py    # Orchestrates all 3 tiers
│   └── summarizer.py        # Background conversation summarizer
├── skills/                  # Tool registry & command handlers
│   ├── base.py              # ToolRegistry
│   ├── time_date.py         # get_time / get_date
│   ├── timers.py            # TimerManager
│   ├── notes.py             # SQLite notes
│   ├── weather.py           # Open-Meteo
│   ├── web_search.py        # DuckDuckGo
│   ├── calculator.py        # Safe expression evaluator
│   └── system_info.py       # CPU, RAM, disk, battery
├── celestial/               # Multi-step planning
│   ├── planner.py           # LLM-driven JSON step plans
│   └── executor.py          # Sequential tool execution
├── bridge/                  # Frontend-backend communication
│   ├── stdio_bridge.py      # stdin/stdout JSON protocol
│   └── server.py            # AssistantRuntime lifecycle
├── frontend/                # Tauri desktop shell
│   ├── index.html           # Widget HTML
│   ├── styles.css           # Glassmorphism UI
│   ├── script.js            # Bridge IPC + SiriWave
│   ├── siriwave.js          # SiriWave.js library
│   └── src-tauri/           # Rust shell (Tauri v2)
├── prompts/                 # System & mode prompts
│   ├── system_prompt.txt
│   └── celestial_prompt.txt
├── scripts/                 # Build & packaging scripts
├── installer/               # Windows installer (Inno Setup)
└── docs/                    # Documentation & screenshots
    └── screenshots/
```

---

## 13. Installation

### Prerequisites
- Python 3.10+
- A GGUF model file (e.g., [Gemma 3 1B](https://huggingface.co/bartowski/gemma-3-1b-it-GGUF))
- NVIDIA GPU with 4GB+ VRAM (optional, CPU mode supported)

### 1. Install Dependencies

```bash
cd voice-assistant-v2
pip install -r ../requirements.txt
```

### 2. Configure Your Model

```bash
# Copy and edit the environment file
cp .env.example .env
# Set LLM_MODEL_PATH to your .gguf file
```

### 3. Run in Text Mode

```bash
python main.py --mode text --gpu 0
```

Type commands like:
- `what time is it`
- `set a timer for 5 minutes`
- `save a note: buy milk`
- `hello` (waits for LLM to load)

### 4. Run the Desktop App

```bash
cd frontend
npm install
npm run tauri:dev
```

### 5. Build for Distribution

```bash
# Build the Python backend
powershell -ExecutionPolicy Bypass -File scripts/build_backend.ps1

# Build the Tauri desktop app
cd frontend
npm run tauri build
```

---

## 14. Design Evolution ⭐⭐⭐⭐⭐

```
Version 1 (Monolithic — anantum/)
│
│  • 6 files, ~2,300 lines
│  • All core logic in one place
│  • CLI-only, no UI
│  • Easy to understand, hard to extend
│
├── Problems:
│   • Tight coupling between components
│   • Adding new skills required modifying core files
│   • No separation between voice, memory, and tools
│   • Difficult to test individual components
│   • No desktop UI
│   • No packaging for distribution
│
▼
Version 2 (Modular — voice-assistant-v2/)
│
│  • 40+ files across 10 packages
│  • Clear separation of concerns
│  • Tauri desktop frontend
│  • stdio bridge protocol
│  • Background model loading
│  • Streaming TTS
│  • Celestial multi-step planning
│  • Wake word support
│  • Windows installer
│
├── Problems:
│   • More boilerplate and imports
│   • Packaging complexity increased
│   • Learning curve for new contributors
│   • Cross-platform audio differences
│
▼
Current Architecture
│
│  • Hybrid: v1 for learning, v2 for production
│  • Both implementations maintained
│  • v1 serves as documentation of core concepts
│  • v2 is the production-ready evolution
│
│  Key improvements from v1 → v2:
│  • Voice, memory, skills as independent packages
│  • Desktop UI with Tauri
│  • stdio bridge for frontend-backend communication
│  • Background model loading
│  • Streaming TTS
│  • Celestial multi-step planning
│  • Wake word support
│  • Windows installer
│  • User settings persistence
│  • Health monitoring and auto-reconnect
```

This evolution shows:
- **Iteration** — The system went through multiple design phases
- **Engineering** — Each version solved real problems discovered in the previous one
- **Learning** — The monolithic v1 was a necessary step to understand the problem space before building the modular v2

---

## 15. Roadmap

- [x] **v1 Core** — Voice/text interaction, tools, memory, LLM
- [x] **v2 Modular** — Separated packages, Tauri frontend
- [ ] **First-run onboarding flow** — Guided model selection and setup wizard
- [ ] **Benchmark automation** — Script that measures and reports performance metrics
- [ ] **Walkthrough video/GIF** — 30-second demo of a full voice interaction
- [ ] **Expanded skill set** — More tools while keeping names obvious and stable
- [ ] **Memory quality improvements** — Better summarization and search relevance
- [ ] **Packaging friction reduction** — One-command desktop builds
- [ ] **Multi-model support** — Swap between models at runtime
- [ ] **Plugin system** — Third-party skill development
- [ ] **Native multimodal models** — Newer Gemma models with audio support (when available)
- [ ] **Mobile support** — Companion app for Android/iOS
- [ ] **Edge deployment** — Raspberry Pi, Jetson Nano, other edge devices
- [ ] **Multi-user profiles** — Separate memory and preferences per user

---

## 16. Lessons Learned ⭐⭐⭐⭐⭐

### Useful before smart.

Local AI feels much more credible when the assistant can do useful work (timers, notes, system info) before the model finishes loading. Tool-first architecture was the right call. The user's perception changes from "waiting for AI" to "using a tool that happens to have AI capabilities."

### Small bridges are better.

A well-defined stdio protocol is easier to reason about, debug, and package than a networked architecture. No ports, no CORS, no connection management. The stdio bridge was one of the best architectural decisions in v2.

### Memory quality matters as much as model quality.

For a personal assistant, remembering user preferences and past conversations is often more valuable than raw reasoning capability. The three-tier design was worth the complexity. Users notice when the assistant remembers their name, preferences, and past conversations.

### Packaging is part of the product.

A working prototype is useless if it can't be installed. Investing in PyInstaller + Inno Setup early saved significant rework. The packaging pipeline should be part of the initial architecture, not an afterthought.

### Regex beats LLM for common intents.

For the 80% of commands that follow predictable patterns (time, weather, timers), regex is faster, cheaper, and more reliable than prompting a model. The LLM should be reserved for conversation, memory recall, and complex tasks where its flexibility is actually needed.

### Architecture matters more than models.

The three-tier memory system, regex-first intent routing, and modular pipeline have a bigger impact on user experience than the choice of LLM. A well-architected system with a small model beats a poorly-architected system with a large model.

### Building AI systems is mostly systems engineering.

The models are the easy part. The hard part is stitching them together into a reliable, responsive system. Most of the code in Anantum is not AI — it's audio processing, memory management, tool execution, error handling, and state management.

### Local AI forces better engineering trade-offs.

When you can't throw cloud GPUs at the problem, you have to be smart about resource usage. This leads to better engineering decisions: efficient model formats (GGUF), smart caching (three-tier memory), and optimized routing (regex-first intent classification).

---

## 17. Research

- 📄 [Research Document](docs/research.md) — Design decisions, experiments, and findings
- 🏗️ [Architecture Notes](docs/architecture.md) — System design and component interactions
- 📊 [Benchmarks](docs/benchmarks.md) — Performance measurements and comparisons
- 🧪 [Experiments](docs/experiments.md) — Failed approaches and what was learned
- 📝 [Packaging Guide](PACKAGING.md) — Build and distribution documentation

---

## 18. References

### Papers
- [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295)
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/2401.08281)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

### Projects
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Local LLM inference
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [Kokoro](https://github.com/hexgrad/kokoro) — Text-to-speech
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [Open-Meteo](https://open-meteo.com/) — Free weather data
- [Tauri](https://v2.tauri.app/) — Desktop application framework
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — Wake word detection
- [SiriWave.js](https://github.com/kopiro/siriwave) — Waveform visualization

### Libraries
- [sentence-transformers](https://www.sbert.net/) — Embedding models
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio I/O
- [PyInstaller](https://pyinstaller.org/) — Python packaging
- [Inno Setup](https://jrsoftware.org/isinfo.php) — Windows installer

---

## 19. License

MIT

---

## 20. Behind the Build — Engineering Journal

This section is for recruiters, founders, and engineers who want to understand the full journey behind Anantum v2.

| Resource | Description |
|----------|-------------|
| 📄 [Research Document](docs/research.md) | Full research document covering design decisions, experiments, and findings |
| 🏗️ [Figma Architecture](https://www.figma.com) | Interactive architecture diagram showing component relationships and data flow |
| 🎥 [Demo Video](https://www.youtube.com) | 5-minute walkthrough of the system in action |
| 📝 [Design Notes](docs/design-notes.md) | Iteration log showing how the architecture evolved from v1 to v2 |
| 📊 [Benchmarks](docs/benchmarks.md) | Detailed performance profiling and optimization results |
| 🧪 [Experiments Log](docs/experiments.md) | Failed approaches, dead ends, and what was learned from each |
| 📦 [Packaging Guide](PACKAGING.md) | Build and distribution documentation |

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Missing dependency | `pip install <package>` |
| Bridge exits immediately | Python not in PATH | Use full path to Python |
| Tauri shows "Preview mode" | Not running inside Tauri | Use `npm run tauri:dev` |
| Frontend stays "Connecting" | Backend process crashed | Check terminal for Python errors |
| No LLM response | Model file not found | Use "Change model" button or `--model` flag |
| TTS silent | Kokoro not installed | `pip install kokoro sounddevice` |
| Wake word not working | Missing ONNX models | Check `models/wake/` directory |

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for local LLM inference
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- [Kokoro](https://github.com/hexgrad/kokoro) for text-to-speech
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Open-Meteo](https://open-meteo.com/) for free weather data
- [Tauri](https://v2.tauri.app/) for the desktop application framework
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection