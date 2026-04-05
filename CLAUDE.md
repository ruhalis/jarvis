# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a local-first voice assistant running on an NVIDIA Jetson Orin Nano Super (8 GB RAM). It supports English and Russian, combining wake word detection, STT, command routing, cloud LLM reasoning, and TTS in a single pipeline.

## Architecture

```
Mic → openWakeWord → Silero VAD → WhisperTRT (STT)
                                        │
                                   transcribed text
                                        │
                                        ▼
                               ┌─ Command Router ─┐
                               │  fuzzy match YAML │
                               └──┬───────────┬───┘
                            match │           │ no match
                                  ▼           ▼
                          Routine Engine    Claude Code CLI
                               │               │
                               └──────┬────────┘
                                      ▼
                              Kokoro / Silero TTS → Speaker
```

- **LLM Backend**: Claude Code CLI (`claude -p`) invoked as subprocess from Python. Authenticated via OAuth (`claude login`), billed against Claude Max subscription — no API key needed.
- **Command Router**: `rapidfuzz.fuzz.partial_ratio()` threshold 80 against YAML routines. Matches execute instantly; misses go to Claude Code CLI. Routines can use `defer_to_llm` to hand off complex parts.
- **Custom Tools**: MCP server (`jarvis_mcp_server.py`) exposes `speak`, `ha_control`, `ha_query`, `set_timer`, `get_weather`, `create_routine`. Connected via `--mcp-config`.
- **State machine**: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE. GPU serialized — one model at a time.
- **Event bus**: Redis Pub/Sub (channels: `wake_detected`, `stt_result`, `tts_request`, `tts_done`, `state_change`, `timer_set`).
- **Process management**: supervisord runs all services.
- **Web dashboard**: FastAPI + HTMX.
- **TTS**: Kokoro TTS for English, Silero V5 for Russian. CosyVoice2 voice cloning planned for Phase 5.
- **Safety**: PreToolUse hooks in `.claude/settings.json` block dangerous HA actions without confirmation.

## Key Design Decisions

- **`speak` as a tool, not streamed text**: The LLM decides what gets spoken. Internal reasoning stays silent. Only `speak()` calls produce audio.
- **CLAUDE.md as single source of truth**: Claude Code reads `~/.jarvis/CLAUDE.md` automatically every turn. Personality, memory, tool docs, HA entities — all in one file.
- **Self-updating memory**: Claude Code uses built-in Read/Write tools to update the Memory section of `~/.jarvis/CLAUDE.md`.
- **Session persistence**: `--session-id jarvis-YYYY-MM-DD` keeps context within a day, rotated at midnight.
- **No API billing**: Claude Code CLI uses Max subscription via OAuth — no ANTHROPIC_API_KEY needed.

## File Structure

```
~/.jarvis/
├── CLAUDE.md                    # personality + memory + tool docs + HA entities
├── config.yaml                  # HA endpoint, TTS config, session settings
├── mcp_config.json              # MCP server config for Claude Code CLI
├── .claude/
│   └── settings.json            # Claude Code hooks (safety gate)
├── routines/
│   ├── default.yaml             # built-in routines
│   └── custom.yaml              # LLM-created routines
├── logs/
│   └── YYYY-MM-DD.jsonl         # daily JSONL transcripts
├── cache/
│   └── tts/                     # pre-generated common phrases (WAV)
└── services/
    ├── jarvis_brain.py           # Claude Code CLI orchestrator
    ├── jarvis_mcp_server.py      # MCP server exposing all custom tools
    ├── jarvis_voice.py           # STT + wake word + VAD pipeline
    ├── jarvis_tts.py             # TTS service (Kokoro/Silero/CosyVoice)
    ├── jarvis_router.py          # Command router + routine engine
    ├── jarvis_dashboard.py       # FastAPI + HTMX web dashboard
    ├── jarvis_timers.py          # Timer/reminder service
    └── safety_gate.py            # PreToolUse hook script
```

## Target Hardware

- **Board**: NVIDIA Jetson Orin Nano Super Developer Kit (8 GB RAM)
- **Software**: JetPack 6.2 (Ubuntu 22.04, CUDA 12.6, TensorRT 10, cuDNN 9)

## Development Environment

- Python 3.10, virtual environment in `.venv`
- Activate: `source .venv/bin/activate`
- Node.js 22+ required (Claude Code CLI runtime)
- Redis required for event bus
- Claude Code authenticated via `claude login` (Max subscription)

## Dependencies

- Claude Code CLI (`@anthropic-ai/claude-code`) — LLM brain
- `mcp` — MCP server framework for custom tools
- `rapidfuzz` — fuzzy matching for command router
- `aiohttp` — Home Assistant API client
- `pyyaml` — routine YAML parsing
- Redis — event bus

## Environment Variables

```bash
# No ANTHROPIC_API_KEY needed — Claude Code uses OAuth subscription auth
HA_URL              # Home Assistant endpoint (http://192.168.1.x:8123)
HA_TOKEN            # Home Assistant long-lived access token
JARVIS_HOME         # defaults to ~/.jarvis
```

## Build Phases

1. **Phase 1** — Minimal voice loop: text input → `claude -p` → speak tool → TTS → speaker
2. **Phase 2** — Full wake-to-speak loop: wake word → VAD → STT → router → brain → TTS
3. **Phase 3** — Home Assistant integration: ha_control, ha_query, timers, weather
4. **Phase 4** — Memory and personality: self-updating CLAUDE.md, daily rotation, routines, dashboard
5. **Phase 5** — Polish: CosyVoice2 voice cloning, web_search, prompt injection defense, robot arm
