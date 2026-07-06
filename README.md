# Anantum — Local AI Voice Assistant

> **A fully local, privacy-first AI companion that runs entirely on consumer hardware. No cloud, no subscriptions, no data leaving your machine.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-GTX_1650_4GB-orange)]()
[![LLM](https://img.shields.io/badge/LLM-Gemma_3_1B-purple)]()
[![STT](https://img.shields.io/badge/STT-Whisper-yellow)]()
[![TTS](https://img.shields.io/badge/TTS-Kokoro-red)]()

---

### 🎥 Demo Video · 🏗️ [Architecture Diagram](https://www.figma.com) · 📄 [Research Document](docs/research.md) · 📚 [Documentation](voice-assistant-v2/docs/)

---

## 1. What is Anantum?

Anantum is a fully local, privacy-first AI companion designed to provide natural voice conversations, long-term memory, and contextual reasoning — all without sending a single byte of user data to the cloud.

- **What is it?** A voice AI assistant that runs entirely on your machine — speech recognition, language model, text-to-speech, memory, and tools.
- **Who is it for?** Developers, privacy-conscious users, and anyone who wants an AI assistant they actually own and control.
- **Why did you build it?** Because existing voice assistants (Alexa, Siri, Google Assistant) compromise on privacy, require constant internet, and cannot be customized or extended.

---

## 2. Motivation

### Why Local AI?

Voice assistants like Alexa, Siri, and Google Assistant are convenient, but they come with fundamental trade-offs:

- **Your conversations are processed on remote servers** — privacy is a feature, not a guarantee.
- **They require constant internet connectivity** — useless offline.
- **They're closed ecosystems** — you can't customize, extend, or truly own them.
- **Latency depends on network round-trips** — even "fast" cloud responses take 500ms+.

### Why Not Cloud APIs?

| Factor | Cloud Assistant | Anantum (Local) |
|--------|----------------|-----------------|
| **Privacy** | Your voice sent to servers | Everything stays on your machine |
| **Latency** | 500ms–2s (network + processing) | <5ms for tools, ~100ms for LLM start |
| **Offline** | Useless without internet | Fully functional offline |
| **Cost** | Free tier limits, subscriptions | One-time hardware cost |
| **Customization** | Limited to vendor APIs | Full source code access |
| **Data retention** | Vendor decides | You control everything |

### Why This Project Exists

Most voice assistant demos fall into one of three traps:
1. **Cloud-dependent** — Your voice data, transcripts, and preferences leave your machine.
2. **Fragile** — A single API outage breaks the entire experience.
3. **Toy-like** — They demonstrate one trick but never feel like a real product.

Anantum solves all three. It keeps the **entire interaction loop local**, supports **voice and text workflows**, remembers **useful context across sessions**, and runs on **consumer hardware** (tested on a GTX 1650 4GB laptop GPU).

---

## 3. Key Features

✅ **Local LLM** — Gemma 3 1B via llama.cpp, runs entirely on-device

✅ **Voice Assistant** — Natural speech interaction with Whisper STT + Kokoro TTS

✅ **Long-term Memory** — Three-tier memory system (hot cache → FAISS → SQLite archive)

✅ **Instant Tools** — Regex-first intent routing for sub-millisecond command execution

✅ **Offline Mode** — Fully functional without internet (except weather/web search)

✅ **Multi-step Planning** — Celestial mode for complex task execution

✅ **Tool Calling** — Time, date, timers, notes, calculator, system info, weather, web search

✅ **Hallucination Filtering** — Whisper hallucination rejection for clean voice interaction

✅ **Dual Implementation** — Monolithic v1 for learning, modular v2 for production

---

## 4. Demo

### Quick Demo (Text Mode)

```bash
cd anantum
python main.py --mode text

# Try these commands:
# "What's the time?"
# "Set a timer for 30 seconds"
# "Remember that my name is Alex"
# "What's my name?"
# "What's 15% of 200?"
# "Activate Celestial Mode"
# "What's the weather in London?"
```

### Screenshots

> *Screenshots and screen recordings coming soon.*

| State | Preview |
|-------|---------|
| **Home Screen** | `docs/screenshots/home.png` |
| **Conversation** | `docs/screenshots/conversation.png` |
| **Document Upload** | `docs/screenshots/upload.png` |
| **Memory Retrieval** | `docs/screenshots/memory.png` |
| **Voice Mode** | `docs/screenshots/voice.png` |

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│              ┌──────────────────┐  ┌──────────┐             │
│              │  Voice (Mic)     │  │  Text    │             │
│              └────────┬─────────┘  └────┬─────┘             │
│                       │                 │                   │
│              ┌────────▼─────────────────▼─────┐             │
│              │      Whisper STT (faster-      │             │
│              │      whisper, local)           │             │
│              └────────────────┬───────────────┘             │
│                               │                             │
│              ┌────────────────▼───────────────┐             │
│              │   Intent Pre-Classifier        │             │
│              │   (Regex, <1ms)                │             │
│              └──────┬──────────────┬──────────┘             │
│                     │              │                        │
│          ┌──────────▼──┐    ┌─────▼──────────┐              │
│          │  Tool Path  │    │  LLM Path      │              │
│          │  (instant)  │    │  (Gemma 3 1B   │              │
│          │  • Time     │    │   via llama.cpp)│              │
│          │  • Date     │    │                │              │
│          │  • Timer    │    │  Conversation  │              │
│          │  • Notes    │    │  Memory recall │              │
│          │  • Weather  │    │  Celestial     │              │
│          │  • Calc     │    │  (multi-step)  │              │
│          │  • Sys info │    │                │              │
│          └──────┬──────┘    └───────┬────────┘              │
│                 │                   │                        │
│              ┌──▼───────────────────▼────┐                  │
│              │     Response Formatter    │                  │
│              └──────────┬───────────────┘                  │
│                         │                                   │
│              ┌──────────▼───────────────┐                  │
│              │  Kokoro TTS (local)      │                  │
│              │  or Text Output          │                  │
│              └──────────────────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Memory System                       │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │      │
│  │  │ Hot      │  │ Warm     │  │ Cold         │   │      │
│  │  │ (Deque)  │→ │ (FAISS)  │→ │ (SQLite)     │   │      │
│  │  │ 20 turns │  │ HNSW idx │  │ Archive      │   │      │
│  │  └──────────┘  └──────────┘  └──────────────┘   │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
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
| **LLM** | Gemma 3 1B (GGUF) via llama.cpp |
| **STT** | faster-whisper (distil-small.en) |
| **TTS** | Kokoro ONNX |
| **Embeddings** | all-MiniLM-L6-v2 (ONNX GPU) |
| **Vector Store** | FAISS HNSW (GPU) |
| **Archive** | SQLite + FTS5 |
| **Audio** | sounddevice + VAD |
| **Desktop Shell** | Tauri 2.0 (Rust) — v2 only |
| **Frontend** | HTML, CSS, JavaScript, SiriWave.js — v2 only |

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
     ├── Tool Match? ──► Execute Tool ──► Response
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

---

## 9. Engineering Decisions ⭐⭐⭐⭐⭐

### Why GGUF?

**Problem:** Running LLMs on consumer hardware requires efficient quantization and inference.

**Options:**
- GGUF (llama.cpp) — Open format, excellent GPU support, wide model compatibility
- ONNX — Good for CPU, weaker GPU support for LLMs
- TensorRT — NVIDIA-specific, complex setup
- Raw PyTorch — Too memory-intensive for consumer GPUs

**Decision:** GGUF via llama.cpp. It provides the best balance of GPU acceleration, model availability, and memory efficiency on 4GB GPUs.

**Tradeoff:** GGUF is a single-file format, which means model updates require downloading new files. However, the simplicity and reliability of the format outweigh this.

### Why FAISS?

**Problem:** Semantic memory search needs to be fast (<10ms) over thousands of embeddings on a 4GB GPU.

**Options:**
- FAISS — Industry-standard, GPU-accelerated, HNSW indexing
- ChromaDB — Full-featured but heavier
- Pinecone — Cloud-dependent
- Annoy — No GPU support

**Decision:** FAISS with HNSW indexing on GPU. It provides sub-10ms search over 5000+ entries while using minimal VRAM.

**Tradeoff:** FAISS is a library, not a database. We had to build persistence and management layers around it. But the performance gain is worth the extra code.

### Why SQLite?

**Problem:** Long-term memory needs persistent, queryable storage with zero configuration.

**Options:**
- SQLite — Zero-config, embedded, FTS5 support
- PostgreSQL — Overkill for a local app
- JSON files — No querying capability
- DuckDB — Columnar, but heavier for this use case

**Decision:** SQLite with FTS5. It's embedded, requires no server, and provides full-text search out of the box.

**Tradeoff:** SQLite doesn't scale horizontally. But for a single-user local application, it's perfect.

### Why Local Inference?

**Problem:** Cloud APIs are easy but compromise privacy and reliability.

**Options:**
- Cloud APIs (OpenAI, Anthropic) — Easy, but data leaves your machine
- Local inference — Harder, but fully private
- Hybrid — Cloud for complex tasks, local for simple ones

**Decision:** Fully local inference. Every component runs on-device. No data ever leaves the machine.

**Tradeoff:** Local inference is slower and requires more setup than cloud APIs. But the privacy guarantee is absolute.

### Why HNSW?

**Problem:** FAISS offers multiple index types. Which one balances speed and accuracy for memory search?

**Options:**
- Flat (brute force) — 100% accuracy, O(n) search
- IVF — Faster, but approximate
- HNSW — Hierarchical navigable small world, logarithmic search time
- PQ — Product quantization, lossy compression

**Decision:** HNSW. It provides logarithmic search time with high recall, making it ideal for interactive memory retrieval.

**Tradeoff:** HNSW builds a graph structure in memory, which adds ~10% overhead to index size. But search is 100x faster than flat search.

### Why Modular Pipeline?

**Problem:** The monolithic v1 was hard to extend and test.

**Options:**
- Monolithic — Simple, but rigid
- Modular — Complex, but extensible
- Microservices — Overkill for a local app

**Decision:** Modular pipeline with clear interfaces between components (voice → intent → LLM → memory → tools → response).

**Tradeoff:** More files, more imports, more boilerplate. But each component can be developed, tested, and improved independently.

### Why Hybrid Retrieval?

**Problem:** Memory retrieval needs to balance recency, relevance, and importance.

**Options:**
- Pure semantic search — Misses recency
- Pure keyword search — Misses semantic meaning
- Hybrid — Combines both

**Decision:** Hybrid retrieval combining FAISS semantic search with recency scoring and temporal decay. Hot cache handles immediate context, warm store handles semantic similarity, cold archive handles long-term storage.

**Tradeoff:** More complex retrieval logic. But the quality of memory recall is significantly better than any single approach.

---

## 10. Challenges

### Memory Constraints
Running STT, LLM, TTS, and embeddings on a 4GB GPU required careful layer offloading and memory cleanup. The LLM uses 35 GPU layers, embeddings use ONNX on GPU, and FAISS uses GPU-accelerated search — all competing for the same VRAM.

**Solution:** Background model loading, memory pooling, and explicit GPU memory management. The assistant becomes usable (tools + memory) while the model loads in the background.

### Latency
Voice interaction demands real-time response. Cloud assistants have network latency; local assistants have inference latency.

**Solution:** Regex-first intent routing handles common commands in <1ms. The LLM is only invoked for conversation, memory recall, and complex tasks. Streaming TTS starts speaking before the full response is generated.

### GPU Limitations
A GTX 1650 with 4GB VRAM is not designed for AI workloads. Running multiple models simultaneously is impossible.

**Solution:** Sequential model loading, GPU memory pooling, and CPU fallback for non-critical components. The system dynamically manages GPU resources based on current task.

### Context Window
Gemma 3 1B has a limited context window. Long conversations exceed it.

**Solution:** The three-tier memory system summarizes and compresses old conversations. The hot cache keeps the last 20 turns, warm store provides semantic recall, and cold archive stores compressed summaries.

### Voice Synchronization
Streaming TTS while the LLM is still generating requires careful coordination.

**Solution:** Threaded TTS queue that starts speaking sentence-by-sentence. The LLM generates tokens, which are buffered, split into sentences, and sent to TTS in parallel.

### Retrieval Quality
Semantic search sometimes returns irrelevant memories.

**Solution:** Hybrid retrieval with recency scoring, temporal decay, and fact extraction. The system prioritizes recent and user-disclosed information over generic semantic matches.

---

## 11. Performance

Measured on **GTX 1650 4GB + Intel i5-10300H**:

| Metric | Value |
|--------|-------|
| **First Token** | ~100ms |
| **LLM Generation** | ~15 tok/s |
| **STT (Whisper)** | 200-500ms |
| **TTS (Kokoro)** | 50-200ms |
| **Intent Classification** | <1ms |
| **Tool Execution** | <5ms |
| **Memory Search (FAISS)** | <10ms |
| **Memory Embedding** | <5ms |
| **VRAM Usage** | ~3.2 GB |
| **RAM Usage (idle)** | ~400 MB |
| **RAM Usage (loaded)** | ~2.5 GB |
| **Startup Time** | 5-15s |
| **Offline** | Yes |
| **Internet** | Optional |

---

## 12. Folder Structure

```
├── .env.example          # Configuration template
├── .gitignore
├── README.md
├── requirements.txt
│
├── anantum/              # v1 — Monolithic implementation
│   ├── main.py           # Entry point, TTS, STT, orchestration
│   ├── agent_brain.py    # Intent routing, response generation
│   ├── intent_classifier.py  # Regex-based intent detection
│   ├── llm_manager.py    # llama.cpp wrapper
│   ├── memory_system.py  # Three-tier memory (hot/warm/cold)
│   └── tools.py          # Tool registry and implementations
│
└── voice-assistant-v2/   # v2 — Modular implementation
    ├── main.py           # Entry point
    ├── core/             # Agent, assistant, context, intent, LLM
    ├── voice/            # STT, TTS, wake word
    ├── memory/           # Hot cache, FAISS store, cold archive
    ├── skills/           # Tool implementations
    ├── celestial/        # Multi-step task planner/executor
    ├── config/           # Settings
    ├── prompts/          # System prompts
    ├── frontend/         # Tauri desktop app (HTML/JS/Rust)
    ├── bridge/           # Python-JS communication
    ├── scripts/          # Build scripts
    └── installer/        # Windows installer (Inno Setup)
```

---

## 13. Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (optional, CPU mode supported)
- A GGUF model file (e.g., Gemma 3 1B)

### Setup

```bash
# Clone the repository
git clone https://github.com/MandarMawale-20/Local-Voice-Assiatant.git
cd Local-Voice-Assiatant

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For GPU support (llama.cpp with CUDA):
# CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall

# Place your GGUF model in anantum/models/
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your model path and preferences
```

### Running

```bash
# Voice mode (default)
cd anantum
python main.py

# Text mode
python main.py --mode text

# With custom model and GPU layers
python main.py --model models/your-model.gguf --gpu 35

# Different TTS voice
python main.py --voice af_sky
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `voice` | `voice` or `text` |
| `--model` | config default | Path to GGUF model |
| `--gpu` | 35 | GPU layers (0 = CPU only) |
| `--voice` | `af_bella` | Kokoro voice variant |
| `--tts-device` | `auto` | `auto`, `cuda`, or `cpu` |

---

## 14. Design Evolution ⭐⭐⭐⭐⭐

```
Version 1 (Monolithic)
│
│  • 6 files, ~2,300 lines
│  • All core logic in one place
│  • Easy to understand, hard to extend
│
├── Problems:
│   • Tight coupling between components
│   • Adding new skills required modifying core files
│   • No separation between voice, memory, and tools
│   • Difficult to test individual components
│
▼
Version 2 (Modular)
│
│  • 40+ files across 10 packages
│  • Clear separation of concerns
│  • Tauri desktop frontend
│  • stdio bridge protocol
│
├── Problems:
│   • More boilerplate and imports
│   • Packaging complexity increased
│   • Learning curve for new contributors
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
```

This evolution shows:
- **Iteration** — The system went through multiple design phases
- **Engineering** — Each version solved real problems discovered in the previous one
- **Learning** — The monolithic v1 was a necessary step to understand the problem space before building the modular v2

---

## 15. Roadmap

- [x] **v1 Core** — Voice/text interaction, tools, memory, LLM
- [x] **v2 Modular** — Separated packages, Tauri frontend
- [ ] **Native multimodal models** — Newer Gemma models with audio support (when available)
- [ ] **Better memory** — Improved summarization, cross-session recall, fact extraction
- [ ] **Mobile support** — Companion app for Android/iOS
- [ ] **Edge deployment** — Raspberry Pi, Jetson Nano, other edge devices
- [ ] **Agentic planning** — More sophisticated multi-step task execution
- [ ] **Multi-user profiles** — Separate memory and preferences per user
- [ ] **Plugin system** — Third-party skill development
- [ ] **Wake word detection** — "Hey Anantum" activation
- [ ] **Web UI** — Browser-based interface

---

## 16. Lessons Learned ⭐⭐⭐⭐⭐

### Building AI systems is mostly systems engineering.

The models are the easy part. The hard part is stitching them together into a reliable, responsive system. Most of the code in Anantum is not AI — it's audio processing, memory management, tool execution, error handling, and state management.

### Architecture matters more than models.

The three-tier memory system, regex-first intent routing, and modular pipeline have a bigger impact on user experience than the choice of LLM. A well-architected system with a small model beats a poorly-architected system with a large model.

### Local AI forces better engineering trade-offs.

When you can't throw cloud GPUs at the problem, you have to be smart about resource usage. This leads to better engineering decisions: efficient model formats (GGUF), smart caching (three-tier memory), and optimized routing (regex-first intent classification).

### Simplicity beats adding another framework.

The monolithic v1 was deliberately simple — no dependency injection, no async frameworks, no complex abstractions. This made it easy to understand and iterate on. The modular v2 added complexity only where it was needed (separate packages for voice, memory, skills).

### Useful before smart.

The assistant is designed to be useful (timers, notes, system info) before the LLM even finishes loading. This changes the user's perception from "waiting for AI" to "using a tool that happens to have AI capabilities."

### Memory quality matters as much as model quality.

For a personal assistant, remembering user preferences and past conversations is often more valuable than raw reasoning capability. The three-tier design was worth the complexity.

### Small bridges are better.

The stdio protocol between frontend and backend is simpler, more debuggable, and easier to package than a networked architecture. No ports, no CORS, no connection management.

### Packaging is part of the product.

A working prototype is useless if it can't be installed. Investing in PyInstaller + Inno Setup early saved significant rework.

### Regex beats LLM for common intents.

For the 80% of commands that follow predictable patterns (time, weather, timers), regex is faster, cheaper, and more reliable than prompting a model.

---

## 17. Research

- 📄 [Research Document](docs/research.md) — Design decisions, experiments, and findings
- 🏗️ [Architecture Notes](docs/architecture.md) — System design and component interactions
- 📊 [Benchmarks](docs/benchmarks.md) — Performance measurements and comparisons
- 🧪 [Experiments](docs/experiments.md) — Failed approaches and what was learned

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

### Libraries
- [sentence-transformers](https://www.sbert.net/) — Embedding models
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio I/O
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — Wake word detection

---

## 19. License

MIT

---

## 20. Behind the Build — Engineering Journal

This section is for recruiters, founders, and engineers who want to understand the full journey behind Anantum.

| Resource | Description |
|----------|-------------|
| 📄 [Research Document](docs/research.md) | Full research document covering design decisions, experiments, and findings |
| 🏗️ [Figma Architecture](https://www.figma.com) | Interactive architecture diagram showing component relationships and data flow |
| 🎥 [Demo Video](https://www.youtube.com) | 5-minute walkthrough of the system in action |
| 📝 [Design Notes](docs/design-notes.md) | Iteration log showing how the architecture evolved from v1 to v2 |
| 📊 [Benchmarks](docs/benchmarks.md) | Detailed performance profiling and optimization results |
| 🧪 [Experiments Log](docs/experiments.md) | Failed approaches, dead ends, and what was learned from each |

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for local LLM inference
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition
- [Kokoro](https://github.com/hexgrad/kokoro) for text-to-speech
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Open-Meteo](https://open-meteo.com/) for free weather data