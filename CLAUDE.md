# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a local-first voice assistant designed to run on a resource-constrained device (8 GB RAM, GPU shared across models). It supports English and Russian, combining wake word detection, speech-to-text, command routing, LLM reasoning, and text-to-speech in a single pipeline.

## Architecture

The pipeline flows: **ReSpeaker Mic → openWakeWord → Silero VAD → WhisperTRT (STT) → Command Router → TTS → Speaker**

- **Command Router** fuzzy-matches transcribed text against a YAML routines file. Matches execute instantly; misses go to an LLM agent (Phi-4-mini or Qwen2.5-3B via Ollama).
- **State machine**: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE. GPU access is serialized — only one model uses the GPU at a time.
- **Event bus**: Redis Pub/Sub (channels: `wake_detected`, `stt_result`, `llm_response`, `tts_done`, `state_change`).
- **Process management**: supervisord runs all services.
- **Web dashboard**: FastAPI + HTMX.
- **Tools/Actions**: Home Assistant (REST/WebSocket), robot arm (LeRobot/USB serial), cloud LLM fallback (Claude API / OpenAI API), timers, calendar, web search.
- **TTS**: Kokoro TTS for English, Silero V5 for Russian.

## Target Hardware

- **Board**: NVIDIA Jetson Orin Nano Super Developer Kit (8 GB RAM)
- **Software**: JetPack 6.2 (Ubuntu 22.04, CUDA 12.6, TensorRT 10, cuDNN 9)

## Development Environment

- Python 3.10, virtual environment in `.venv`
- Activate: `source .venv/bin/activate`
