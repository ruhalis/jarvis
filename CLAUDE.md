# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Authoritative spec: `TECHNICAL.md` (v2.0). Display co-processor spec: `RGB-MATRIX.md`. This file is the short orientation; defer to those for details.

## Project Overview

Jarvis is a voice assistant running on an NVIDIA Jetson Orin Nano Super (8 GB RAM). **English-only** (Russian was dropped in v2.0 to free memory for a local LLM tier). Wake word → streaming STT → command router → dual-mode brain → TTS → speaker, with an optional RGB matrix display co-processor for status/animations.

## Architecture

```
Mic → openWakeWord → Silero VAD → Parakeet-TDT-0.6b-v3 (streaming STT)
                                        │
                                   transcribed text (partials + final)
                                        │
                                        ▼
                               ┌─ Command Router ─┐
                               │  fuzzy match YAML │
                               └──┬───────────┬───┘
                            match │           │ no match
                                  ▼           ▼
                          Routine Engine    Brain (dual-mode)
                               │               │
                               └──────┬────────┘
                                      ▼
                              Kokoro / CV3 TTS → Speaker
                                      │
                                      └─► state_change → jarvis_display.py → MCU → HUB75 panel
```

## Dual-Mode Brain

Per-turn mode selection in `jarvis_brain.py`:

- **Mode A — Cloud (default):** Claude Agent SDK (Python, `anthropic` package), Haiku 4.5 default with escalation to Sonnet 4.6. Prompt caching is mandatory. Auth: `ANTHROPIC_API_KEY` (per-token billing — **not** Max OAuth).
- **Mode B — Local fallback:** `claw-code` (vendored Rust harness at `claw-code/`) → `llama-server --jinja` → **Gemma 4 E2B** Q4_K_M. Triggered when API key missing, network down, or Mode A fails 3× in a row. Optional vision via `mmproj-gemma4-e2b-f16.gguf` for a `look_and_answer` MCP tool.

Both modes share the same `jarvis_mcp_server.py`, `~/.jarvis/CLAUDE.md`, and Redis schema. Session state is owned by the brain (`~/.jarvis/sessions/YYYY-MM-DD.json`), so mode flips are transparent to the conversation.

## Components

- **STT:** Parakeet-TDT-0.6b-v3 streaming via NeMo (TensorRT export where supported). Emits `stt_partial` + `stt_result`. Pre-warms the brain on stable partials.
- **TTS:** Kokoro (always hot, ~150 MB) is the daily driver. Fun-CosyVoice3-0.5B-2512 (~3.5 GB) loads on-demand for the cloned Jarvis voice (Phase 5). CV3 + Gemma cannot both be hot.
- **Command Router:** `rapidfuzz.fuzz.partial_ratio()` ≥ 80 against YAML routines. `defer_to_llm` hands complex parts to the brain.
- **Custom tools (MCP):** `speak`, `ha_control`, `ha_query`, `set_timer`, `get_weather`, `web_search`, `create_routine`, `memory_write`, optional `look_and_answer` (Mode B vision).
- **Safety:** PreToolUse hooks in `~/.jarvis/.claude/settings.json` block dangerous HA actions (locks, alarms, garage) without verbal confirmation.
- **Event bus:** Redis Pub/Sub — `wake_detected`, `stt_partial`, `stt_result`, `tts_request`, `tts_done`, `state_change`, `timer_set`, `brain_state`, `tts_state`.
- **State machine:** IDLE → LISTENING → PROCESSING → SPEAKING → IDLE.
- **Display co-processor:** `jarvis_display.py` subscribes to `state_change` + TTS levels and forwards JSON over USB serial to an Interstate 75 / ESP32-S3 driving the HUB75 panel. Do **not** drive HUB75 from the Jetson directly (no real-time, no maintained driver).
- **Process management:** supervisord. `llama-server` stays resident so Gemma's ~1-min warmup is paid once at boot.
- **Web dashboard:** FastAPI + HTMX — conversation log, mode indicator (ONLINE/OFFLINE), token + cost meter, system status.

## Key Design Decisions

- **`speak` is a tool, not streamed text.** Internal reasoning stays silent; only `speak()` calls produce audio. Critical for Mode B where Gemma is more prone to thinking out loud.
- **CLAUDE.md is the single source of truth** read by both adapters. Personality, memory, tool docs, HA entities live in `~/.jarvis/CLAUDE.md`. A slimmer variant is used for Mode B (Gemma's 2048-token context).
- **Self-updating memory** via a `memory_write` MCP tool that appends to the Memory section, gated by the safety hook.
- **Local session storage** (not Anthropic's hosted sessions, not claw-code's session feature) — keeps mode switching transparent and debuggable.
- **English-only.** Enables smaller/faster TTS and makes a 5B local LLM tier feasible on 8 GB.

## Memory Budget (8 GB total)

| Configuration | Resident | Headroom |
|---|---|---|
| Mode A + Kokoro | ~2.0 GB | ~6.0 GB |
| Mode A + CV3 swapped in | ~5.3 GB | ~2.7 GB |
| Mode B text-only (Gemma + Kokoro) | ~5.0 GB | ~3.0 GB |
| Mode B with vision (+ mmproj) | ~5.5 GB | ~2.5 GB |
| **Mode B + CV3 hot — FORBIDDEN** | ~8.3 GB | <0 GB |

TTS service must refuse CV3 swap-in while Mode B is active. Configure 8 GB swap as a safety net during model loads.

## File Structure

```
~/.jarvis/
├── CLAUDE.md                    # personality + memory + tool docs + HA entities
├── config.yaml                  # HA endpoint, TTS config, display config
├── mcp_config.json              # MCP server config (absolute path to venv python!)
├── claw.json                    # Mode B harness config (claw-code → llama-server)
├── .claude/
│   └── settings.json            # PreToolUse safety hooks
├── routines/
│   ├── default.yaml
│   └── custom.yaml              # LLM-created routines
├── sessions/
│   └── YYYY-MM-DD.json          # per-day message list (shared by both adapters)
├── logs/
│   └── YYYY-MM-DD.jsonl         # per-turn transcript with mode + token usage
├── cache/
│   └── tts/                     # pre-rendered common phrases (WAV)
└── services/
    ├── jarvis_brain.py          # dual-mode dispatcher (CloudAdapter + LocalAdapter)
    ├── jarvis_mcp_server.py     # all custom tools
    ├── jarvis_voice.py          # wake word + VAD + Parakeet streaming STT
    ├── jarvis_tts.py            # Kokoro + CV3 swap, publishes RMS levels
    ├── jarvis_router.py         # fuzzy router + routine engine
    ├── jarvis_dashboard.py
    ├── jarvis_timers.py
    ├── jarvis_display.py        # Redis → USB serial bridge to HUB75 MCU
    ├── safety_gate.py           # PreToolUse hook script
    └── firmware/jarvis_matrix/  # MCU firmware (Interstate 75 or ESP32-S3)
```

## Target Hardware

- **Board:** NVIDIA Jetson Orin Nano Super Developer Kit (8 GB)
- **Software:** JetPack 6.2 (Ubuntu 22.04, CUDA 12.6, TensorRT 10, cuDNN 9)
- **Display (optional):** Waveshare RGB-Matrix-P3 64×64 + Interstate 75 (RP2040/RP2350) or ESP32-S3 + HUB75 adapter, 5 V / 4 A dedicated PSU.

## Development Environment

- Python 3.10 in `.venv` (`source .venv/bin/activate`)
- Rust toolchain (for `claw-code/rust && cargo build --release --workspace`)
- `llama.cpp` built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`, **or** the prebuilt `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` container (text-only)
- Redis required for the event bus
- supervisord for process management

## Dependencies

- `anthropic` — Claude Agent SDK (Mode A)
- `claw-code` (vendored at `claw-code/`) — Mode B harness
- llama.cpp + Gemma 4 E2B Q4_K_M GGUF (~3 GB) + optional `mmproj-gemma4-e2b-f16.gguf` (~0.5 GB)
- `mcp`, `rapidfuzz`, `aiohttp`, `pyyaml`
- NeMo (Parakeet) + TensorRT export tooling
- Kokoro; later Triton + TensorRT-LLM for CV3

## Environment Variables

```bash
ANTHROPIC_API_KEY        # Mode A; absence forces Mode B
HA_URL                   # http://192.168.1.x:8123
HA_TOKEN                 # long-lived access token
JARVIS_HOME              # defaults to ~/.jarvis
LLAMA_SERVER_URL         # http://127.0.0.1:8080/v1
JARVIS_BRAIN_FORCE_MODE  # "cloud" | "local" | "" (auto)
```

## Build Phases (see TECHNICAL.md §8 for the full checklist)

1. **Phase 1** — Mode A only: Agent SDK + MCP `speak` tool + JSONL logging + prompt caching verified.
2. **Phase 2** — Voice pipeline: openWakeWord + Silero VAD + Parakeet streaming, command router, partial pre-warm, barge-in.
3. **Phase 3** — Home Assistant: `ha_control`, `ha_query`, safety hook, timers, weather.
4. **Phase 4** — Memory + **Mode B**: self-updating CLAUDE.md, daily rotation, dashboard, build claw-code, build llama.cpp + Gemma 4 E2B, verify claw-code ↔ `llama-server --jinja` MCP round-trip, ModeSupervisor + failover, optional `look_and_answer` vision tool.
5. **Phase 5** — Polish: Fun-CosyVoice3-0.5B-2512 (Triton + TensorRT-LLM), `web_search`, prompt-injection defense on tool results, RGB matrix integration end-to-end, robot arm.

## Critical Gotchas

- **MCP `command`** must be the absolute path to the venv's `python`, not `python3` — Claude Code/Agent SDK doesn't inherit your shell's PATH.
- **Prompt caching is mandatory in Mode A** — without `cache_control` on CLAUDE.md + tool defs, monthly cost rises 3–5×.
- **`llama-server --jinja` is mandatory in Mode B** — without it Gemma 4 emits plain text instead of structured tool calls and `speak` will be dropped.
- **Verify claw-code ↔ `llama-server` `/v1` round-trip early** in Phase 4. If it doesn't work, fork claw-code or write a thin Python harness around `/v1/chat/completions` with tool support.
- **Mode B + CV3 simultaneously is forbidden** — TTS service must check `brain_state` before swapping CV3 in.
- **Don't drive HUB75 from the Jetson.** Use an MCU co-processor over USB serial; see `RGB-MATRIX.md`.
