# Jarvis LLM Layer — Technical Specification

**Version:** 2.0
**Date:** April 2026
**Platform:** NVIDIA Jetson Orin Nano Super Developer Kit (8 GB RAM, ARM64, JetPack/CUDA)
**Status:** Pre-implementation
**Brain:** Dual-mode — Claude Agent SDK (cloud, default) + Gemma 4 E2B via claw-code (local fallback)

---

## 1. Goal

Build the "brain" layer of the Jarvis voice assistant — the component that takes transcribed speech, reasons about it, executes actions (smart home, timers, web search), and produces spoken responses. The brain must feel like a single intelligent agent, not a chatbot wrapper.

**Success criteria:**
- End-to-end voice latency under 3 seconds (wake word → spoken response) for simple queries
- Correct tool execution on first attempt for common smart home commands
- Natural, concise voice responses — 1–3 sentences, never reads lists aloud
- Bilingual operation (English and Russian) with automatic language detection
- Persistent memory across sessions — Jarvis remembers preferences and context
- Graceful degradation when internet is unavailable (YAML routines still work)

---

## 2. Architecture Decision: Dual-Mode Brain

Jarvis runs in one of two modes, selected per-turn by a mode supervisor in `jarvis_brain.py`:

| | Mode A — Cloud (default) | Mode B — Local fallback |
|---|---|---|
| Harness | **Claude Agent SDK** (Python) | **claw-code** Rust binary |
| Model | Claude Haiku 4.5 (escalate to Sonnet 4.6 on hard queries) | **Gemma 4 E2B (5B, VLA)** Q4_K_M via llama.cpp `llama-server` |
| Auth | `ANTHROPIC_API_KEY` | none (local) |
| Cost | ~$5–15/month (with prompt caching) | $0 |
| Latency | ~300–600 ms TTFT | ~1–3 s TTFT on Orin Nano |
| Quality | strong tool-use, reliable | weaker — more clarification, occasional dropped tool calls |
| Internet | required | not required |

**Mode selection logic** (each turn, before dispatch):
- If `ANTHROPIC_API_KEY` present AND network reachable AND last N requests didn't 5xx/429 → **Mode A**
- Otherwise → **Mode B** (announce "running offline, sir" once per outage transition)

Both modes connect to the same `jarvis_mcp_server.py` over MCP, share the same `~/.jarvis/CLAUDE.md`, and emit the same `speak` tool calls to Redis. The voice pipeline is mode-agnostic.

**Why Claude Agent SDK over Claude Code CLI for cloud:**
- First-class prompt caching (90% discount on cached input — CLAUDE.md + tool defs cached per session)
- Python-native — no subprocess JSON streaming, no `--session-id`/`--resume` UUID juggling
- Cleaner MCP wiring directly in process
- **No Max-subscription ToS ambiguity** around programmatic/automated use of a developer tool
- Estimated $5–15/month on Haiku 4.5 with caching for typical personal use, vs $100–200/month flat for Max

**Why claw-code for the local fallback (vs raw Ollama API):**
- claw-code is an OSS Rust port of the Claude Code agent loop — same think → tool → observe pattern
- Speaks MCP natively → reuses `jarvis_mcp_server.py` unchanged
- Routes to OpenAI-compatible endpoints via `.claw.json` config → points at Ollama's `/v1` server
- Avoids hand-rolling a tool-calling loop on top of `ollama.chat()`
- Already vendored at `claw-code/` in this repo (Rust workspace, build with `cargo build --workspace`)

> **Verify before Phase 4:** confirm claw-code's current OpenAI-compatible base-URL config (custom `base_url` + `api_key` in `.claw.json`) actually round-trips MCP tool calls against `llama-server --jinja`. If not, we either patch claw-code or fall back to a thin custom harness around `llama-server`'s `/v1/chat/completions` with tool support.

**Why Gemma 4 E2B for local:**
- 5B-parameter Vision-Language model; ~3.0 GB resident at Q4_K_M — fits alongside Parakeet (0.7 GB) + Kokoro (0.15 GB) + system on the 8 GB Jetson (NVIDIA validates the Q4_K_M build on Jetson Orin Nano Super 8 GB; recommends 8 GB swap as a load-time safety net)
- Native tool-calling enabled via `llama-server --jinja` (Gemma 4's chat template advertises tools to the model) — stronger than Gemma 3 4B at deciding when to call vs. answer
- **Optional vision** via the `mmproj-gemma4-e2b-f16.gguf` projector (~0.5 GB extra). Lets a single `look_and_answer` MCP tool grab a webcam frame so Jarvis can answer "what am I holding?" without a separate VLM service. Skip the mmproj load if vision isn't wanted — the model runs text-only.
- English-strong (matches the English-only TTS scope decision in §3.1.2)
- GGUF weights on Hugging Face (`ggml-org/gemma-4-E2B-it-GGUF`, `unsloth/gemma-4-E2B-it-GGUF`); served by llama.cpp's `llama-server`, which exposes an OpenAI-compatible `/v1` endpoint that claw-code can hit directly
- Fits with CV3 unloaded; CV3 + Gemma cannot both be hot — TTS service swaps Kokoro↔CV3 and brain swaps cloud↔local independently

**Accepted tradeoffs:**
- **Two code paths to maintain.** The brain service has a Mode A path (Agent SDK) and a Mode B path (claw-code subprocess). Mitigated by sharing MCP server, CLAUDE.md, and Redis schema.
- **Local mode quality drop.** Gemma 4 E2B will occasionally fail to emit a `speak` call, mis-route to the wrong tool, or hallucinate entity IDs. The safety-gate hook (§3.5) is the backstop.
- **Local mode latency.** First inference after load is ~1 minute on Orin Nano Super (model warmup); steady-state TTFT is ~1–3 s. Keep `llama-server` resident (`supervisord`) so the warmup is paid once at boot, not per turn.
- Need llama.cpp built from source on the Jetson with CUDA (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`) plus ~3 GB of GGUF weights (and ~0.5 GB more if mmproj/vision is enabled). NVIDIA also publishes a prebuilt Jetson container (`ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`) for text-only deployments.
- claw-code's OpenAI-compat routing against `llama-server` is the integration risk — verify early.

---

## 3. System Components

### 3.1 Voice Pipeline (Context — already designed)

```
Mic → openWakeWord → Silero VAD → Parakeet-TDT-0.6b-v3 (streaming STT)
                                        │
                                   transcribed text (incremental partials + final)
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
                              Kokoro / Silero TTS
                                      │
                                      ▼
                                   Speaker
```

The LLM layer sits at the "Claude Code CLI" box. Everything else is handled by other services communicating via Redis pub/sub.

### 3.1.1 STT: Parakeet-TDT-0.6b-v3 (streaming)

**Chosen model:** NVIDIA `nvidia/parakeet-tdt-0.6b-v3` running in cache-aware streaming mode via NeMo (TensorRT export where supported).

**Why Parakeet over WhisperTRT:**
- **Streaming-native** Token-and-Duration Transducer — emits committed tokens incrementally as audio arrives, rather than re-decoding a sliding window like Whisper. First partial token typically lands in 200–400 ms vs 800–1500 ms for streamed Whisper on Jetson.
- **Multilingual v3** covers 25 European languages including **Russian** — meets the bilingual requirement (v1/v2 were English-only, do not use them).
- **SOTA WER** on multilingual benchmarks at 0.6B params (smaller than whisper-large-v3, more accurate on most European languages).
- **Built-in punctuation, capitalization, and word timestamps** — cleaner input to the LLM, no second pass.
- NVIDIA-native (NeMo) → first-class Jetson + TensorRT path.

**Accepted tradeoffs:**
- ~0.7 GB INT8 / ~1.2 GB FP16 resident with streaming cache — heavier than whisper-base, fits the 8 GB budget.
- Streaming cache-aware TensorRT export from NeMo is less paved than WhisperTRT's prebuilt path; expect up-front deployment work.
- Russian quality is strong on benchmarks but less battle-tested in the wild than Whisper's Russian — validate with real mic audio before committing past Phase 2.

**Latency unlock — partial-transcript pre-warm:**
Because partials are *committed* (don't rewrite), the brain service can act on stable partials before VAD endpoints the utterance:
1. Run the command router's fuzzy match on growing partials — pre-resolve routine matches.
2. **Pre-spawn `claude -p`** as soon as a stable partial arrives, so Node.js / Claude Code CLI cold-start (~500 ms–1 s) overlaps with the user finishing their sentence.
3. Use STT confidence + VAD silence jointly to endpoint earlier.

This is expected to shave 500–1000 ms off perceived wake-to-speak latency on top of the raw STT speedup.

### 3.1.2 TTS: Kokoro (default) + Fun-CosyVoice3-0.5B (cloned voice)

**Scope decision:** Jarvis is **English-only**. Russian support is dropped. This unlocks the smaller/faster variant of every component and makes a local LLM tier feasible.

**Default TTS — Kokoro:**
- 82M params, ~150 MB GPU resident, Apache 2.0
- High MOS for its size, ~50 prebuilt voices — pick one as the canonical Jarvis voice
- ~100–200 ms first-audio on Jetson; effectively free latency-wise
- **Always hot.** Handles all routine responses.

**Premium TTS — Fun-CosyVoice3-0.5B-2512 (Phase 5):**
- Hugging Face: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (paper: arXiv 2505.17589)
- 0.5B params, ~3.5 GB GPU resident
- **Bi-streaming** (text-in + audio-out streaming) → 150 ms first-audio
- **Zero-shot voice cloning** from a 6–10 s reference clip; 0.81 % CER, 77.4 % speaker similarity — best-in-class at this size
- 9 languages incl. English (Russian not supported, irrelevant here) + 18 dialects, cross-lingual cloning
- Optimized path is **Triton + TensorRT-LLM** (RTF 0.04–0.10, ~4× speedup over plain PyTorch). Without TRT-LLM, falls back to roughly CosyVoice2 speed.

**Why CV3 over CosyVoice2:** same param count, better metrics across the board (CER, speaker similarity, latency), bi-streaming built in, more recent (Dec 2025).

**Why CV3 in addition to Kokoro instead of replacing it:** CV3 is ~23× larger and only earns its keep when you actually want a *cloned* voice. For "turn on the lights — done, sir" you don't need 3.5 GB of model. Kokoro is the daily driver; CV3 is the special-occasion engine.

**Swap pattern (TTS service, `jarvis_tts.py`):**
- Subscribe to `tts_request` Redis channel; payload includes `voice` field
- `voice = "kokoro:<voice_id>"` → render with Kokoro (always hot)
- `voice = "cv3:<speaker_ref>"` → if CV3 not loaded, swap it in (~1–2 s); render; keep loaded for ~60 s of inactivity then unload
- Pre-cache common phrases as WAV under `~/.jarvis/cache/tts/` — bypasses both engines for instant playback

**Accepted tradeoffs:**
- CV3 first use after idle pays a 1–2 s warmup. Mitigation: pre-warm on `wake_detected` if last response used CV3.
- CV3 + a future local LLM (Gemma 4 E2B, ~3.5 GB) cannot both be hot — they must serialize against each other on the 8 GB board.
- Triton + TensorRT-LLM on Jetson Orin Nano ARM64 is non-trivial to deploy; budget real time for it in Phase 5.
- December 2025 model — less battle-tested than CosyVoice2; validate audio quality with real reference clips before committing.

### 3.2 Brain Service — Dual-Mode Dispatcher

**Entry point:** A Python service (`jarvis_brain.py`) that:
1. Subscribes to Redis channel `stt_result`
2. Picks the mode for this turn (cloud vs local)
3. Dispatches to the appropriate adapter
4. Routes `speak` tool calls to Redis channel `tts_request` (same downstream for both modes)
5. Logs all events to JSONL transcript

```python
# jarvis_brain.py — simplified dispatcher
class Brain:
    def __init__(self):
        self.cloud = CloudAdapter()   # Claude Agent SDK
        self.local = LocalAdapter()   # claw-code → Ollama → Gemma
        self.health = ModeSupervisor()

    async def handle(self, user_text: str, session_id: str):
        mode = self.health.pick_mode()
        adapter = self.cloud if mode == "cloud" else self.local
        try:
            async for event in adapter.stream(user_text, session_id):
                if is_speak_tool_call(event):
                    await redis.publish("tts_request", extract_speech(event))
                log_to_jsonl(event, mode=mode)
        except (NetworkError, RateLimitError, ProviderError) as e:
            self.health.record_failure(mode, e)
            if mode == "cloud":
                # immediate failover for this turn
                await self.local.stream(user_text, session_id)
```

#### 3.2.1 Mode A — CloudAdapter (Claude Agent SDK)

```python
from anthropic import AsyncAnthropic
from anthropic.lib.tools import MCPClient

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY

async def stream(user_text: str, session_id: str):
    system = load_claude_md()  # ~/.jarvis/CLAUDE.md
    async with client.messages.stream(
        model="claude-haiku-4-5",     # default; escalate to claude-sonnet-4-6 on demand
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},  # 90% discount on reuse
        }],
        messages=load_session(session_id) + [{"role": "user", "content": user_text}],
        mcp_servers=[{"type": "stdio", "command": MCP_SERVER_CMD}],
        max_tokens=512,
    ) as stream:
        async for event in stream:
            yield event
```

Key points:
- **Prompt caching is mandatory** — without `cache_control` on CLAUDE.md + tool defs, monthly cost rises 3–5×.
- **MCP server is the same** `jarvis_mcp_server.py` used by Mode B.
- **Session storage** lives in `~/.jarvis/sessions/<date>.json` (a list of message dicts), not in Anthropic's session feature.
- **Model escalation:** if Haiku returns a low-confidence response (e.g., the model asks "did you mean…" or fails to emit a tool call), retry once on Sonnet 4.6. Cap escalations per day to avoid bill spikes.

#### 3.2.2 Mode B — LocalAdapter (claw-code → llama.cpp `llama-server` → Gemma 4 E2B)

```python
async def stream(user_text: str, session_id: str):
    proc = await asyncio.create_subprocess_exec(
        CLAW_BIN,                    # /opt/jarvis/claw-code/rust/target/release/claw
        "prompt", user_text,
        "--config", str(CLAW_CONFIG),  # ~/.jarvis/claw.json
        "--mcp-config", str(MCP_CONFIG_PATH),
        "--session", session_id,
        "--stream", "json",
        stdout=asyncio.subprocess.PIPE,
        cwd=str(JARVIS_HOME),
        env={**os.environ, "OPENAI_API_KEY": "sk-local"},  # llama-server ignores key, claw-code requires non-empty
    )
    async for line in proc.stdout:
        yield json.loads(line)
```

`~/.jarvis/claw.json` (Mode B harness config):

```json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "gemma-4-E2B-it",
  "system_prompt_path": "~/.jarvis/CLAUDE.md",
  "tool_choice": "auto",
  "max_tokens": 384
}
```

`llama-server` is started by supervisord at boot and stays resident, so Gemma's ~1-minute warmup is paid once. Recommended launch flags (from NVIDIA's Jetson Orin Nano guide):

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-gemma4-e2b-f16.gguf \   # omit for text-only (saves ~0.5 GB)
  -c 2048 \
  --image-min-tokens 70 --image-max-tokens 70 \
  --ubatch-size 512 --batch-size 512 \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 --flash-attn on \
  --no-mmproj-offload --jinja -np 1
```

`--jinja` is mandatory — it activates Gemma 4's native tool-calling chat template, without which claw-code will get plain text back instead of structured tool calls.

#### 3.2.3 Mode supervisor

```python
class ModeSupervisor:
    def __init__(self):
        self.recent_failures = deque(maxlen=5)
        self.last_announced_mode = None

    def pick_mode(self) -> str:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "local"
        if not self._network_up():
            return "local"
        if sum(self.recent_failures) >= 3:
            return "local"   # cooldown
        return "cloud"

    def _network_up(self) -> bool:
        # cheap TCP check to api.anthropic.com:443, 500ms timeout
        ...
```

When the supervisor flips modes, it emits a `state_change` Redis event so the dashboard reflects "ONLINE" vs "OFFLINE" and the brain can prepend "running offline, sir — " to the next response (once, not on every turn).

### 3.3 CLAUDE.md — The Single Source of Truth

Location: `~/.jarvis/CLAUDE.md`

Claude Code reads this file automatically from the working directory (`cwd`) on every turn. It replaces the OpenClaw pattern of SOUL.md + MEMORY.md + SKILL.md.

**Structure:**

```markdown
# Jarvis — Voice Assistant

## Identity
You are Jarvis, a personal AI voice assistant.
Personality: understated British wit, quiet confidence, dry humor.
Use "sir" sparingly — once per conversation, not every sentence.
Never break character. You are not Claude, you are Jarvis.

## Voice Rules
- ALWAYS respond using the `speak` tool. Never output bare text as a response.
- Keep spoken responses to 1–3 sentences unless the user asks for detail.
- Match the user's language: Russian input → respond in Russian.
- For lists: summarize counts ("You have 5 items") — never enumerate aloud.
- For complex answers: speak the summary, log detail to the dashboard.
- Acknowledge commands instantly: "Right away, sir" → then execute.

## Tools Available
- `speak` — say something to the user (REQUIRED for all responses)
- `ha_control` — control Home Assistant entities (lights, climate, media)
- `ha_query` — read Home Assistant sensor states
- `set_timer` — set a countdown timer
- `get_weather` — get current weather and forecast
- `web_search` — search the web for information
- `create_routine` — save a new voice routine to YAML

## Memory
- User: Arlan
- Location: Almaty, Kazakhstan (UTC+6)
- Languages: English, Russian (auto-detect from input)
- Preferences: thermostat default 22°C, wake-up 7:00 AM

## Home Assistant
- Endpoint: http://192.168.1.x:8123
- light.living_room, light.bedroom, light.kitchen, light.hallway
- climate.main_thermostat (supports: set_temperature, turn_on, turn_off)
- media_player.living_room_tv (supports: turn_on, turn_off, volume_set)
- sensor.outdoor_temperature, sensor.indoor_humidity

## Behavioral Boundaries
- NEVER execute lock/unlock, alarm arm/disarm, or garage door without
  verbal confirmation from the user.
- For purchases or financial actions: always refuse, suggest the user
  do it manually.
- If uncertain about a command, ask for clarification rather than guessing.
```

**Self-updating memory:** Claude Code includes built-in `Read` and `Write` file tools. When the user says "remember that I prefer 23 degrees," Jarvis can append to the Memory section of CLAUDE.md using the Write tool. This persists across sessions.

### 3.4 Custom Tools via MCP Servers

Tools are implemented as MCP (Model Context Protocol) servers. Claude Code connects to them via an `mcp_config.json` file passed with `--mcp-config`.

**MCP config:** `~/.jarvis/mcp_config.json`

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["/absolute/path/to/jarvis/services/jarvis_mcp_server.py"],
      "env": {
        "HA_URL": "http://192.168.1.x:8123",
        "HA_TOKEN": "eyJ..."
      }
    }
  }
}
```

**Note:** use the absolute path to the venv's `python` binary, not `python3`. Claude Code spawns the MCP server without inheriting your shell's PATH/venv activation, so a bare `python3` will miss packages installed in the venv.

The MCP server (`jarvis_mcp_server.py`) exposes all custom tools:

#### 3.4.1 `speak` — Voice Output (Required)

```python
@server.tool("speak")
async def speak(text: str, language: str = "en") -> str:
    """Say something to the user via the speaker."""
    await redis.publish("tts_request", json.dumps({
        "text": text,
        "lang": language,
        "priority": "normal"
    }))
    return f"[Spoken]: {text}"
```

Why a tool and not streamed text: The LLM decides what gets spoken. Internal reasoning, tool planning, and observation text stay silent. Only explicit `speak()` calls produce audio. This eliminates the need for `<think>/<speak>` tag parsing.

#### 3.4.2 `ha_control` — Home Assistant Actions

```python
@server.tool("ha_control")
async def ha_control(entity_id: str, service: str, attributes: dict = {}) -> str:
    """Control a Home Assistant entity (lights, climate, media)."""
    domain = entity_id.split(".")[0]
    result = await ha_client.call_service(domain, service,
        entity_id=entity_id, **attributes)
    return f"Done: {service} on {entity_id}"
```

#### 3.4.3 `ha_query` — Read Sensor States

```python
@server.tool("ha_query")
async def ha_query(entity_id: str) -> str:
    """Get the current state of a Home Assistant entity."""
    state = await ha_client.get_state(entity_id)
    unit = state["attributes"].get("unit_of_measurement", "")
    return f"{entity_id}: {state['state']} {unit}"
```

#### 3.4.4 `set_timer` — Timers and Reminders

```python
@server.tool("set_timer")
async def set_timer(duration_seconds: int, label: str) -> str:
    """Set a countdown timer or reminder."""
    timer_id = str(uuid.uuid4())[:8]
    await redis.publish("timer_set", json.dumps({
        "id": timer_id,
        "duration": duration_seconds,
        "label": label
    }))
    return f"Timer '{label}' set for {duration_seconds}s (id: {timer_id})"
```

#### 3.4.5 `get_weather` — Weather Information

```python
@server.tool("get_weather")
async def get_weather(location: str = "Almaty") -> str:
    """Get current weather and forecast for a location."""
    data = await weather_api.get(location)
    return (f"Weather in {data['city']}: {data['temp']}°C, {data['condition']}. "
            f"Forecast: high {data['high']}°C, low {data['low']}°C.")
```

#### 3.4.6 `create_routine` — Dynamic Routine Creation

```python
@server.tool("create_routine")
async def create_routine(trigger_phrase: str, actions: list) -> str:
    """Save a new voice routine the user defined."""
    routine = {"trigger": trigger_phrase, "actions": actions}
    routines_path = Path("~/.jarvis/routines/custom.yaml").expanduser()
    with open(routines_path, "a") as f:
        yaml.dump([routine], f)
    return f"Routine saved: '{trigger_phrase}' → {len(actions)} actions"
```

### 3.5 Hooks — Safety and Routing

Claude Code hooks are configured in `~/.jarvis/.claude/settings.json`. They run shell commands before/after tool use.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__jarvis__ha_control",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.jarvis/services/safety_gate.py"
          }
        ]
      }
    ]
  }
}
```

**Safety gate script** (`safety_gate.py`):

```python
#!/usr/bin/env python3
import json, sys

# Claude Code passes tool input via stdin
input_data = json.loads(sys.stdin.read())
tool_input = input_data.get("tool_input", {})
service = tool_input.get("service", "")

DANGEROUS = ["lock", "unlock", "alarm_arm", "alarm_disarm", "open_cover"]

if any(d in service for d in DANGEROUS):
    # Output JSON to block the tool call
    print(json.dumps({
        "decision": "block",
        "reason": f"Safety: '{service}' requires voice confirmation. Ask the user to confirm."
    }))
else:
    print(json.dumps({"decision": "approve"}))
```

### 3.6 Session and Memory Management

**Session storage:** the brain owns session state directly — neither adapter relies on a hosted session feature. Each turn's messages are appended to `~/.jarvis/sessions/YYYY-MM-DD.json` (a list of `{role, content}` dicts). Both adapters load this list and prepend it to the request, which keeps mode switches transparent — Gemma sees the same conversation Claude did.

**Daily rotation:** at midnight, a maintenance task runs one final Mode A turn (preferred) asking Claude to summarize the day's important facts and write them via the `memory_write` MCP tool into CLAUDE.md's Memory section. Then it starts a fresh empty session file for the next day.

**Why not hosted sessions:** Anthropic's session API and claw-code's session feature both exist, but using either ties us to one mode. Local JSON keeps mode switching cheap and debuggable.

**JSONL transcript:** Every message (user input, assistant response, tool calls, tool results) is appended to `~/.jarvis/logs/YYYY-MM-DD.jsonl`. One JSON object per line. Human-readable, greppable.

```json
{"ts": "2026-04-04T10:15:00Z", "type": "user", "text": "Turn on the lights"}
{"ts": "2026-04-04T10:15:01Z", "type": "tool_call", "tool": "ha_control", "args": {"entity_id": "light.living_room", "service": "turn_on"}}
{"ts": "2026-04-04T10:15:01Z", "type": "tool_result", "tool": "ha_control", "result": "Done"}
{"ts": "2026-04-04T10:15:02Z", "type": "tool_call", "tool": "speak", "args": {"text": "Living room lights are on, sir."}}
```

### 3.7 Stream Filtering

Both adapters yield events in a normalized shape (`{type, ...}`) so the filter is mode-agnostic:

| Event type | Action |
|---|---|
| Assistant text | Log to dashboard + JSONL. Do NOT send to TTS. |
| Tool use (`speak`) | Extract text, publish to `tts_request` Redis channel |
| Tool use (`ha_control`, etc.) | Log to dashboard, tool executes via MCP server |
| Tool result | Log to JSONL |
| Message end | Update state machine to IDLE or SPEAKING |

Mode A (Agent SDK) emits these events natively. Mode B normalizes claw-code's stream-json into the same shape inside `LocalAdapter.stream()`.

Key insight: because `speak` is a tool, there is a clean separation. All spoken output goes through the tool. All other text is internal reasoning that gets logged but never vocalized — this matters more in Mode B, where Gemma is more prone to "thinking out loud."

---

## 4. Command Router (Fast Path)

Before any request reaches Claude Code CLI, the command router does a fuzzy match against YAML-defined routines. This is the fast path — no LLM, no API call, no latency.

**File:** `~/.jarvis/routines/default.yaml`

```yaml
routines:
  - trigger: ["good morning", "let's get started", "wake up"]
    actions:
      - ha_control: { entity_id: light.living_room, service: turn_on, attributes: { brightness: 200 } }
      - ha_control: { entity_id: light.kitchen, service: turn_on }
      - speak: { text: "Good morning, sir. Lights are on." }
      - defer_to_llm: { prompt: "Give a brief morning briefing: weather, calendar, news" }

  - trigger: ["good night", "go to sleep"]
    actions:
      - ha_control: { entity_id: light.living_room, service: turn_off }
      - ha_control: { entity_id: light.bedroom, service: turn_off }
      - ha_control: { entity_id: climate.main_thermostat, service: set_temperature, attributes: { temperature: 19 } }
      - speak: { text: "Good night, sir. Lights off, thermostat set to 19." }

  - trigger: ["stop", "shut up", "be quiet", "cancel"]
    actions:
      - system: { command: stop_tts }
      - speak: { text: "" }  # empty = silence
```

**Matching logic:** `rapidfuzz.fuzz.partial_ratio()` with threshold 80. If score >= 80, execute the routine directly. If no match, forward to Claude Code CLI.

**`defer_to_llm` action:** A routine can partially execute (lights on) and then hand off to the LLM for the complex part (morning briefing). This is the bridge between fast path and smart path.

---

## 5. File Structure

```
~/.jarvis/
├── CLAUDE.md                    # personality + memory + tool docs + HA entities
├── config.yaml                  # HA endpoint, TTS config, session settings
├── mcp_config.json              # MCP server config for Claude Code CLI
├── .claude/
│   └── settings.json            # Claude Code hooks (safety gate, etc.)
├── routines/
│   ├── default.yaml             # built-in routines
│   └── custom.yaml              # LLM-created routines
├── logs/
│   ├── 2026-04-04.jsonl         # today's transcript
│   └── 2026-04-03.jsonl         # yesterday's transcript
├── cache/
│   ├── tts/                     # pre-generated common phrases (WAV)
│   └── speaker_embedding.pt     # CosyVoice cached voice embedding
└── services/
    ├── jarvis_brain.py           # Claude Code CLI orchestrator (this spec)
    ├── jarvis_mcp_server.py      # MCP server exposing all custom tools
    ├── jarvis_voice.py           # STT + wake word + VAD pipeline
    ├── jarvis_tts.py             # TTS service (Kokoro/Silero/CosyVoice)
    ├── jarvis_router.py          # Command router + routine engine
    ├── jarvis_dashboard.py       # FastAPI + HTMX web dashboard
    ├── jarvis_timers.py          # Timer/reminder service
    └── safety_gate.py            # PreToolUse hook for dangerous HA actions
```

---

## 6. Dependencies

### Required on Jetson

| Package | Purpose | Install |
|---|---|---|
| `anthropic` (Python SDK) | Mode A — Claude Agent SDK | `pip install anthropic` |
| Rust toolchain | Build claw-code (Mode B harness) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| claw-code | Mode B agent harness (vendored at `claw-code/`) | `cd claw-code/rust && cargo build --release --workspace` |
| llama.cpp (CUDA) | Local model server (Mode B) | Build from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`, or pull `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` |
| Gemma 4 E2B (Q4_K_M GGUF) | Local model weights, ~3 GB | `huggingface-cli download ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf` |
| `mmproj-gemma4-e2b-f16.gguf` | Vision projector (optional, for `look_and_answer` tool), ~0.5 GB | `huggingface-cli download ggml-org/gemma-4-E2B-it-GGUF mmproj-gemma4-e2b-f16.gguf` |
| Python 3.10+ | All services | Pre-installed on JetPack |
| Redis | Event bus | `apt install redis-server` |
| `mcp` | MCP server framework | `pip install mcp` |
| `rapidfuzz` | Fuzzy match for command router | `pip install rapidfuzz` |
| `aiohttp` | Home Assistant API client | `pip install aiohttp` |
| `pyyaml` | Routine YAML parsing | `pip install pyyaml` |

### Authentication

```bash
# Mode A — Anthropic API key (per-token billing, no Max subscription)
export ANTHROPIC_API_KEY="sk-ant-..."

# Mode B — no auth required (local llama-server on 127.0.0.1:8080)
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Mode A; absence forces Mode B
export HA_URL="http://192.168.1.x:8123"
export HA_TOKEN="eyJ..."
export JARVIS_HOME="$HOME/.jarvis"
export LLAMA_SERVER_URL="http://127.0.0.1:8080/v1"  # Mode B endpoint (llama.cpp llama-server)
export JARVIS_BRAIN_FORCE_MODE=""         # "cloud" | "local" | "" (auto)
```

---

## 7. Memory Budget (Updated)

```
OS + system services        ~0.7 GB
Parakeet-TDT-0.6b-v3 (INT8) ~0.7 GB   (streaming, cache-aware)
Kokoro TTS (English)        ~0.15 GB  (default, always hot)
Fun-CosyVoice3-0.5B         ~3.5 GB   (Phase 5, on-demand swap-in for cloned voice)
Gemma 4 E2B Q4_K_M (llama.cpp) ~3.0 GB (Mode B local brain, hot when no internet)
mmproj-gemma4-e2b-f16        ~0.5 GB   (optional, only if vision/`look_and_answer` enabled)
openWakeWord + Silero VAD   ~0.05 GB
Redis + orchestrator        ~0.1 GB
claw-code (Rust binary)     ~0.05 GB  (only resident when Mode B active)
MCP server (Python)         ~0.05 GB
Python services             ~0.15 GB
───────────────────────────────────────
Mode A, Kokoro hot                   ~1.95 GB    headroom ~6.0 GB
Mode A, CV3 swapped in               ~5.30 GB    headroom ~2.7 GB
Mode B text-only (Gemma + Kokoro)    ~4.95 GB    headroom ~3.0 GB
Mode B with vision (+ mmproj)        ~5.45 GB    headroom ~2.5 GB
Mode B (Gemma + CV3 hot)             ~8.30 GB    headroom <0 GB  ← over budget, forbidden
Available                            8.0 GB
```

**Coexistence rules:**
- **Cloud mode (A)** uses no local LLM memory — full headroom for STT + TTS.
- **Local mode (B)** keeps Gemma resident; coexists fine with Kokoro.
- **Gemma 4 E2B + CV3 simultaneously exceeds 8 GB.** If Mode B is active, the TTS service must refuse CV3 swap-in — fall back to Kokoro for the cloned-voice path until cloud comes back, or unload `llama-server` briefly. NVIDIA recommends 8 GB of swap configured on the Jetson as a safety net during model loads.
- The mode supervisor and TTS service publish their resident-model state to Redis (`brain_state`, `tts_state`) so each can see what the other is holding.

---

## 8. Build Phases

### Phase 1 — Minimal voice loop, Mode A only (Week 1–2)
**Goal:** Speak to Jarvis, get a spoken response. Cloud only — local fallback is Phase 4.

- [ ] Install Python `anthropic` SDK on Jetson; set `ANTHROPIC_API_KEY`
- [x] Write `CLAUDE.md` with basic personality and voice rules
- [x] Implement MCP server with `speak` tool
- [x] Write `mcp_config.json` pointing to the MCP server
- [ ] Implement `CloudAdapter` in `jarvis_brain.py` using Agent SDK
- [ ] Wire prompt caching on CLAUDE.md + tool defs (verify cache_read tokens > 0 in usage)
- [x] Wire: hardcoded text input → adapter → `speak` tool → stdout *(TTS deferred to Phase 2)*
- [x] Verify the LLM only speaks via the `speak` tool, never outputs bare text
- [x] Add basic JSONL logging including per-turn `mode`, input/output tokens, cache hit ratio

**Phase 1 gotchas (carried over from CLI prototype):**
- MCP server `command` must be an absolute path to the venv's `python`, not `python3`.
- MCP tools are namespaced as `mcp__<server>__<tool>` — match by suffix.
- Scope `allowed_tools` so the model can't try built-ins it doesn't have.

**Milestone:** Type a sentence, hear Jarvis respond. Cache hit ratio > 80% after the first turn.

### Phase 2 — Voice pipeline integration (Week 3–4)
**Goal:** Full wake-to-speak loop.

- [ ] Connect openWakeWord → Silero VAD → Parakeet-TDT-0.6b-v3 (streaming, cache-aware)
- [ ] Export Parakeet from NeMo to TensorRT INT8 for Jetson Orin Nano; benchmark first-partial latency
- [ ] Publish both partial and final transcripts to Redis (`stt_partial`, `stt_result`)
- [ ] Implement Redis pub/sub bridge: `stt_result` → brain → `tts_request`
- [ ] Pre-warm `claude -p` on stable partial transcripts to overlap Node cold-start with end of utterance
- [ ] Implement command router with `rapidfuzz` and default YAML routines
- [ ] Implement state machine: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
- [ ] Add `defer_to_llm` support in routines
- [ ] Handle barge-in (user speaks while Jarvis is speaking → stop TTS)

**Milestone:** Say "Hey Jarvis, what's the weather?" and hear a spoken answer.

### Phase 3 — Home Assistant integration (Week 5–6)
**Goal:** Control smart home by voice.

- [ ] Add `ha_control` and `ha_query` tools to MCP server
- [ ] Populate CLAUDE.md with real HA entity list (auto-discover from HA API)
- [ ] Implement safety gate hook in `.claude/settings.json`
- [ ] Add `set_timer` tool + timer service
- [ ] Add `get_weather` tool
- [ ] Test multi-step commands: "Turn off all lights and set thermostat to 20"

**Milestone:** "Hey Jarvis, turn on the living room lights" → lights turn on, Jarvis confirms.

### Phase 4 — Memory, personality, and **local fallback (Mode B)** (Week 7–9)
**Goal:** Jarvis remembers, learns, and survives offline.

Memory & personality:
- [ ] Enable self-updating CLAUDE.md (a `memory_write` MCP tool that appends to the Memory section, gated by safety hook)
- [ ] Implement daily session rotation (rotate JSONL transcript + summary write at midnight)
- [ ] Add `create_routine` tool (LLM creates YAML routines from conversation)
- [ ] Tune CLAUDE.md personality through real usage
- [ ] Implement web dashboard (FastAPI + HTMX): conversation log, mode indicator, system status, cost/token meter

Local fallback (Mode B):
- [ ] Build claw-code release binary on Jetson (`cd claw-code/rust && cargo build --release --workspace`)
- [ ] Build llama.cpp with CUDA on Jetson; download `gemma-4-E2B-it-Q4_K_M.gguf` (and optionally `mmproj-gemma4-e2b-f16.gguf` for vision); start `llama-server --jinja`; benchmark TTFT and tools/sec
- [ ] **Verify claw-code can route to `llama-server`'s `/v1` with `.claw.json`** and round-trip MCP tool calls (requires `--jinja` so Gemma 4's tool template is active). If not, fork claw-code or write a thin custom harness around `/v1/chat/completions` with tool support.
- [ ] Implement `LocalAdapter` and `ModeSupervisor` in `jarvis_brain.py`
- [ ] Add network health check + failure cooldown
- [ ] Test forced offline: pull network, verify Mode B kicks in within one turn
- [ ] Test recovery: restore network, verify Mode A resumes
- [ ] Tune Gemma's system prompt (slim CLAUDE.md variant — `llama-server` is launched with `-c 2048` per NVIDIA's reference config, so the local prompt budget is tight; trim CLAUDE.md aggressively for Mode B)
- [ ] (Optional) Wire a `look_and_answer` MCP tool that captures a webcam frame and posts it to `llama-server`'s multimodal endpoint — exercises Gemma 4's VLA capability and gives Mode B a vision tool the cloud path doesn't currently have

> **Skip cron/proactive prompts.** The original plan had scheduled `claude -p` jobs (morning briefing, etc.). Drop them — they add cost on Mode A and unattended hallucinations on Mode B with little user value. Use deterministic routines instead.

**Milestone:** "Jarvis, remember I prefer 23 degrees" → next day, thermostat defaults to 23. Pull the network cable mid-conversation → next response is Gemma, dashboard flips to OFFLINE, no user-facing error.

### Phase 5 — Polish and expansion (Week 9+)
**Goal:** Production quality.

- [ ] Add Fun-CosyVoice3-0.5B-2512 for cloned Jarvis voice (English); deploy via Triton + TensorRT-LLM on Jetson; implement on-demand load/unload around Kokoro
- [ ] Add Telegram as remote control channel (future, not priority)
- [ ] Add `web_search` tool to MCP server
- [ ] Implement prompt injection defense on tool results
- [ ] Performance tuning: measure and optimize end-to-end latency
- [ ] Add offline degradation indicator on dashboard
- [ ] Robot arm integration (LeRobot, SO-ARM100 — future phase)

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Anthropic API outage / rate limit | Mode A unavailable | Mode supervisor flips to Mode B (local Gemma). YAML routines still work for the simple stuff. |
| API cost overrun on Mode A | Bill spike | Prompt caching on CLAUDE.md + tools. Cap monthly spend in Anthropic console. Daily token budget alarm on dashboard. Default to Haiku, escalate to Sonnet sparingly. |
| claw-code OpenAI-compat routing doesn't work as expected | Mode B unbuildable as designed | Fallback plan: write a thin Python harness around `ollama.chat()` with tool support. Validate this in Phase 4 spike before committing. |
| Gemma 4 E2B tool-use brittleness | Mode B drops `speak` calls or hallucinates entities | Launch `llama-server --jinja` so the native tool template is active. Slimmer CLAUDE.md variant for local (2048-token context). Stricter `tool_choice`. Safety gate hook catches dangerous outputs. Prefer YAML routines when offline. |
| Gemma 4 E2B + CV3 both hot → OOM | Jetson swap thrash or kill (combined ~8.3 GB > 8 GB) | Brain and TTS publish resident-model state to Redis; TTS refuses CV3 swap-in while Mode B is active. 8 GB swap configured as safety net during transitions. |
| CLAUDE.md grows too large | Context window overflow (esp. on Gemma) | Daily summary + truncate old turns. Memory section capped (e.g., 2 KB) with LRU eviction. |
| Prompt injection via HA sensor data | Unintended actions | Safety gate hook blocks dangerous actions on both modes. Sanitize all tool results before returning. |
| TTS latency with CosyVoice | Slow voice output | Kokoro is the daily driver. CosyVoice is Phase 5 only. Pre-cache common phrases. |
| MCP server crashes | Tools unavailable in both modes | supervisord restarts MCP server. Both adapters tolerate tool errors. |
| Anthropic pricing changes | Cost shifts | Mode B is the structural hedge — Jarvis keeps working at $0/month if cloud becomes uneconomic. |

---

## 10. Key Architectural Patterns

| Pattern | Source | Our Implementation |
|---|---|---|
| SOUL.md + MEMORY.md | OpenClaw | Combined into `CLAUDE.md` (Claude Code native) |
| SKILL.md tool descriptions | OpenClaw | Tool docs in `CLAUDE.md` + MCP server tool definitions |
| JSONL transcripts | OpenClaw | Append-only daily log files |
| Cron / heartbeat | OpenClaw | Timer service + scheduled `claude -p` queries |
| ReAct agent loop | General | Claude Code CLI built-in |
| Prompt injection defense | OpenClaw lessons | `PreToolUse` hooks via settings.json |
| Context compaction | OpenClaw | Claude Code CLI built-in |
| Skills self-creation | OpenClaw | `create_routine` tool writes YAML |
| Custom tools | MCP standard | MCP server with tool definitions |
