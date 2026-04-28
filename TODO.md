# Jarvis TODO

Authoritative spec: `TECHNICAL.md` (v2.0). Display spec: `RGB-MATRIX.md`. Orientation: `CLAUDE.md`.

Recommended order: **A → B → C → E (in parallel with C) → D8 spike first, then rest of D → F.** The integration risk that matters most is **D8** — prove it before building `LocalAdapter` around it.

---

## Current state (2026-04-28)

**Done:** Redis event bus (`jarvis_bus.py`), state machine (`jarvis_state.py`), MCP server with `speak` (`jarvis_mcp_server.py`), command router with fuzzy match + barge-in (`jarvis_router.py`), brain wired to `claude -p` CLI in daemon + one-shot mode (`jarvis_brain.py`), per-day JSONL logging, default routines + project CLAUDE.md.

**Not done / next up:**

1. **Voice capture pipeline** (B1–B4) — no `jarvis_voice.py` yet; nothing actually emits `wake_detected` / `stt_partial` / `stt_result` outside manual `redis-cli publish`.
2. **Phase 3 HA + safety** (C1–C4) — `ha_control` is a stub in the router, no MCP impls, no `safety_gate.py`, no `jarvis_timers.py`.
3. ~~**Cloud brain on SDK** (A1, A3, A5) — currently shells out to `claude` CLI; revisit when budget/caching tracking matters.~~ **Done (2026-04-28):** ported to `AsyncAnthropic` with cache_control + escalation + JSONL usage logging.
4. ~~**TTS service** — `jarvis_tts.py` (Kokoro consumer of `tts_request`, RMS publisher) is missing.~~ **Done (2026-04-28):** `jarvis/services/jarvis_tts.py` consumes `tts_request` + `barge_in`, publishes `tts_state` / `tts_level` / `tts_done`, with a pluggable engine (`auto` → kokoro / `say` / `espeak` / `null`), WAV cache under `cache/tts/`, and chunked playback via `sounddevice` with mid-utterance barge-in. Smoke-tested locally with the macOS `say` fallback. Kokoro install + voice tuning still TODO on the Jetson.

---

## A. Finish Phase 1 — Mode A (Cloud, Claude Agent SDK)

> **Status (2026-04-28):** Ported to `AsyncAnthropic` SDK. `jarvis/services/jarvis_brain.py` now calls `client.messages.stream(...)` directly with an inline `speak` tool that publishes to Redis `tts_request` (the MCP server stays available for other clients but is off the brain's hot path). `cache_control: ephemeral` on system + tool def, per-turn usage logged, Haiku → Sonnet escalation with daily cap, sessions persisted as `sessions/YYYY-MM-DD.json`.

- [x] **A1.** `anthropic` added to `requirements.txt`; brain refuses to start without `ANTHROPIC_API_KEY`. Per-token billing.
- [x] **A2.** `CloudAdapter` ported to `AsyncAnthropic().messages.stream(...)` — replaces the prior `claude -p` subprocess.
- [x] **A3.** `cache_control: {"type": "ephemeral"}` on system block (CLAUDE.md) + `speak` tool def. Per-turn `cache_read_input_tokens` / `cache_creation_input_tokens` logged to `logs/YYYY-MM-DD.jsonl`. Note: caching only activates above the model's minimum prefix (4096 tokens for Haiku 4.5) — current CLAUDE.md is short, so reads will stay at 0 until the prompt grows or a 1h TTL is set; the wiring is in place.
- [x] **A4.** Per-day `sessions/YYYY-MM-DD.json` (list of `{role, content}`) loaded on each turn, full assistant `content` (incl. `tool_use` blocks) appended for fidelity.
- [x] **A5.** Escalation: if default Haiku turn finishes without calling `speak`, re-run from the pre-turn history on `claude-sonnet-4-6`. Daily cap = 20 in `cache/escalation_state.json`; final fallback emits a canned apology to TTS.

---

## B. Phase 2 — Voice pipeline (Parakeet streaming)

- [ ] **B1.** Build NeMo on the Jetson; pull `nvidia/parakeet-tdt-0.6b-v3`. Run cache-aware streaming demo against the dev mic; measure first-partial latency.
- [ ] **B2.** Export to TensorRT INT8 where supported; fall back to FP16 if the streaming export path bites. Target <400 ms first-partial.
- [ ] **B3.** `services/jarvis_voice.py`: openWakeWord → Silero VAD → Parakeet streaming → publish `stt_partial` (every commit) + `stt_result` (on endpoint) to Redis.
- [ ] **B4.** Pre-warm hook in the brain: on stable `stt_partial`, kick off a speculative Agent SDK request (or claw-code spawn for Mode B) so cold-start overlaps end of utterance. Cancel if final differs materially.
- [x] **B5.** Command router (`jarvis/services/jarvis_router.py`) — rapidfuzz match against `routines/default.yaml`, `defer_to_llm` publishes to `llm_request`. Actions: `speak`, `ha_control` (stub), `system`, `defer_to_llm`.
- [x] **B6.** State machine (`jarvis/services/jarvis_state.py`) + barge-in in router: `wake_detected` while SPEAKING publishes barge-in and transitions to LISTENING. `tts_done` returns to IDLE.

> **Phase 2 status:** Router + state machine + Redis bus (`jarvis_bus.py`) + MCP `speak` tool are in. **Voice capture is NOT** — `jarvis_voice.py` (openWakeWord + Silero VAD + Parakeet streaming) doesn't exist yet. B1–B4 are the remaining blockers.

---

## C. Phase 3 — Home Assistant + safety

- [ ] **C1.** Extend `jarvis_mcp_server.py` with `ha_control`, `ha_query`, `set_timer`, `get_weather` (impls drafted in TECHNICAL.md §3.4).
- [ ] **C2.** Auto-discover entities from HA `/api/states`; render the canonical list into the `## Home Assistant` section of `~/.jarvis/CLAUDE.md` on boot (idempotent regeneration between markers).
- [ ] **C3.** `services/safety_gate.py` + `~/.jarvis/.claude/settings.json` PreToolUse matcher on `mcp__jarvis__ha_control`. Block lock/unlock/alarm/garage; require verbal confirmation flow (set a `pending_confirmation` Redis key; next user "yes" releases it).
- [ ] **C4.** Timer service (`jarvis_timers.py`): consumes `timer_set`, fires `tts_request` on expiry.

---

## D. Phase 4 — Memory + Mode B local fallback (the big one)

- [ ] **D1.** Self-updating memory: add `memory_write` MCP tool that appends a single bullet under the `## Memory` section of `~/.jarvis/CLAUDE.md`. LRU-cap the section at ~2 KB.
- [ ] **D2.** Daily rotation cron at midnight: one final Mode A turn summarizes the day, calls `memory_write`, then opens a fresh `sessions/YYYY-MM-DD.json` and `logs/YYYY-MM-DD.jsonl`.
- [ ] **D3.** Build `claw-code`: `cd claw-code/rust && cargo build --release --workspace`. Confirm the binary path matches what `LocalAdapter` will exec.
- [ ] **D4.** Build `llama.cpp` with CUDA on Jetson: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=ON && cmake --build build --config Release -j4`. Or pull the prebuilt `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` for text-only.
- [ ] **D5.** Download weights: `huggingface-cli download ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf` (+ `mmproj-gemma4-e2b-f16.gguf` if you want vision).
- [ ] **D6.** Add 8 GB swap as a load-time safety net (per NVIDIA's Jetson guide).
- [ ] **D7.** supervisord unit for `llama-server` with the exact flags from TECHNICAL.md §3.2.2 (`-c 2048`, `-ngl 99`, `--flash-attn on`, `--jinja`, `--no-mmproj-offload`, `--ubatch-size 512 --batch-size 512`, `--host 127.0.0.1 --port 8080`). Pay the ~1-min warmup once at boot.
- [ ] **D8. ⚠ Integration spike — verify claw-code ↔ `llama-server`.** Write `~/.jarvis/claw.json` (provider `openai`, `base_url http://127.0.0.1:8080/v1`, model `gemma-4-E2B-it`, `tool_choice auto`); send a test prompt that should call `speak`; confirm a structured tool call comes back over stream-json. **If it doesn't round-trip, fork claw-code or write a thin Python harness around `/v1/chat/completions` with tool support.** Don't move on until this is green.
- [ ] **D9.** Implement `LocalAdapter` (subprocess over claw-code stream-json) and normalize its events into the same shape `CloudAdapter` yields, so `jarvis_brain.py`'s stream filter stays mode-agnostic.
- [ ] **D10.** `ModeSupervisor`: env-key check + cheap TCP probe to `api.anthropic.com:443` (500 ms timeout) + recent-failure deque. Failover within one turn on cloud exception. On flip, publish `state_change`, prepend "running offline, sir — " once per outage transition.
- [ ] **D11.** Slim CLAUDE.md variant for Mode B (Gemma's 2048-token context is tight). Strip examples and long HA entity comments; keep tool list, identity, hard rules.
- [ ] **D12.** Resident-model coordination: brain publishes `brain_state` (`cloud|local|local+vision`); TTS publishes `tts_state` (`kokoro|cv3`). TTS refuses CV3 swap-in while `brain_state` ∈ {`local`, `local+vision`}.
- [ ] **D13.** (Optional) `look_and_answer` MCP tool: capture a webcam frame via OpenCV, POST to `llama-server`'s multimodal endpoint, return text. Mode-B-only — Mode A gets a `(not available in cloud mode)` shim or routes to a vision-capable Anthropic model later.
- [ ] **D14.** Dashboard: FastAPI + HTMX live conversation, mode badge, token + cost meter, tail of `logs/*.jsonl`.

---

## E. RGB Matrix display co-processor

- [ ] **E1.** Bench-wire panel + 5 V/4 A PSU + chosen MCU (Interstate 75 W if you want Wi-Fi; ESP32-S3 + adapter if you want cheaper). 1000 µF cap across the panel's 5 V input.
- [ ] **E2.** Flash MCU firmware in `services/firmware/jarvis_matrix/` with three hard-coded modes (`idle_clock`, `listen_pulse`, `think_spinner`) — verify zero flicker at full brightness before adding the protocol.
- [ ] **E3.** USB-serial line protocol: one JSON object per line — `{"mode":"listen"}`, `{"mode":"clock","t":"14:32"}`, `{"mode":"weather","temp":12,"icon":"rain"}`, `{"mode":"speak","level":0.42}`. Tagged-binary later if throughput matters.
- [ ] **E4.** `services/jarvis_display.py`: subscribe to `state_change`, map → MCU command. Add `display:` block to `config.yaml` (port, baud, brightness_idle, brightness_active).
- [ ] **E5.** Speaking waveform: have `jarvis_tts.py` publish RMS (~30 Hz) on `tts_level`; `jarvis_display.py` forwards to MCU EQ-bar animation.
- [ ] **E6.** Weather + timer overlays once Phase 3 lands. Brightness ramp tied to time of day.

---

## F. Phase 5 — Polish

- [ ] **F1.** CV3: deploy `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` via Triton + TensorRT-LLM on Jetson; on-demand load/unload around Kokoro; pre-warm on `wake_detected` if last response used CV3.
- [ ] **F2.** `web_search` MCP tool. Sanitize tool results before returning (prompt-injection defense — strip prompt-shaped content from sensor strings and search snippets).
- [ ] **F3.** End-to-end latency budget: instrument wake → first-audio; target <3 s for simple queries.
- [ ] **F4.** Robot arm (LeRobot, SO-ARM100) — separate spike, out of scope here.

---

## Note on Agent SDK + Gemma

The Claude Agent SDK speaks Anthropic's Messages API; Gemma 4 E2B via `llama-server` speaks OpenAI's. They don't interop natively. Translation proxies (anthropic-proxy, claude-code-router, LiteLLM) work for plain chat but **tool-use translation leaks** (parallel calls, streaming deltas, `tool_use_id`). Stick with the split: Agent SDK for Mode A, `claw-code` for Mode B, normalized at the adapter boundary in `jarvis_brain.py`.
