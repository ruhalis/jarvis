# Architecture
┌─────────────────────────────────────────────────────┐
│  ALWAYS ON (CPU, <1% usage)                         │
│                                                     │
│  ReSpeaker Mic → openWakeWord → Silero VAD          │
│  (captures audio)  (detects       (detects speech   │
│                     "Hey Jarvis")  start/end)        │
└──────────────────────┬──────────────────────────────┘
                       │ clipped audio buffer
                       ▼
┌─────────────────────────────────────────────────────┐
│  STT (GPU, ~300ms)                                  │
│                                                     │
│  WhisperTRT — small multilingual model (EN + RU)    │
└──────────────────────┬──────────────────────────────┘
                       │ transcribed text
                       ▼
┌─────────────────────────────────────────────────────┐
│  COMMAND ROUTER (CPU, <10ms)                        │
│                                                     │
│  Fuzzy-matches text against routines YAML           │
│                                                     │
│  Match found?                                       │
│  ├── YES → Routine Engine (instant, no GPU)         │
│  └── NO  → LLM Agent (GPU, 2-5 sec)                │
└───────┬─────────────────────────┬───────────────────┘
        │                         │
        ▼                         ▼
┌───────────────┐   ┌─────────────────────────────────┐
│ ROUTINES      │   │ LLM AGENT (GPU)                 │
│               │   │                                 │
│ YAML-defined  │   │ Phi-4-mini / Qwen2.5-3B        │
│ action lists  │   │ via Ollama                      │
│               │   │                                 │
│ "let's get    │   │ Reasons about request,          │
│  started"     │   │ calls tools via function        │
│  → lights on  │   │ calling, handles conversation   │
│  → play music │   │                                 │
│  → briefing   │   │ Can CREATE new routines         │
│               │   │ ("when I say X, do Y")          │
└───────┬───────┘   └──────────┬──────────────────────┘
        │                      │
        └──────────┬───────────┘
                   │ action calls / response text
                   ▼
┌─────────────────────────────────────────────────────┐
│  TOOLS & ACTIONS                                    │
│                                                     │
│  Home Assistant ─── REST/WebSocket API               │
│  (lights, climate, media, sensors)                  │
│                                                     │
│  Robot Arm ──────── LeRobot / USB serial            │
│  (SO-ARM100, future expansion)                      │
│                                                     │
│  Cloud LLM ──────── Claude API / OpenAI API         │
│  (fallback for complex queries)                     │
│                                                     │
│  Other ──────────── timers, calendar, web search    │
└──────────────────────┬──────────────────────────────┘
                       │ response text
                       ▼
┌─────────────────────────────────────────────────────┐
│  TTS (CPU only, ~200ms)                             │
│                                                     │
│  English → Kokoro TTS (82M params)                  │
│  Russian → Silero V5                                │
│  Option → CosyVoice2-0.5B (voice cloning, ~3-4GB)  │
└──────────────────────┬──────────────────────────────┘
                       │ audio
                       ▼
                   [ Speaker ]

## Backbone

Redis Pub/Sub ──── event bus between all components
                   channels: wake_detected, stt_result,
                   llm_response, tts_done, state_change

State Machine ──── IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
                   serializes GPU access (only one model
                   uses GPU at a time)

FastAPI + HTMX ─── web dashboard showing status, conversation
                   history, system resources, config panel

supervisord ────── process manager for all services


## Memory budget 
OS + services          ~0.7 GB
WhisperTRT small       ~1.0 GB
LLM 3.8B Q4           ~2.5 GB
Kokoro + Silero TTS    ~0.2 GB  (or CosyVoice2-0.5B ~3-4 GB)
openWakeWord + VAD     ~0.05 GB
Redis + orchestrator   ~0.1 GB
─────────────────────────────
Total                  ~4.5 GB of 8 GB available