# Jarvis LLM Layer — Technical Specification

**Version:** 1.1
**Date:** April 2026
**Platform:** NVIDIA Jetson Orin Nano Super Developer Kit (8 GB RAM, ARM64, JetPack/CUDA)
**Status:** Pre-implementation

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

## 2. Architecture Decision: Claude Code CLI

**Chosen approach:** Use Claude Code CLI (`claude`) as the sole LLM backend, invoked as a subprocess from Python. Billed against the user's Claude Max subscription — no separate API billing.

**What Claude Code CLI provides for free:**
- Agentic tool-calling loop (think → call tool → observe result → repeat)
- MCP server support for custom tools
- CLAUDE.md auto-loading into system prompt every turn (from `cwd`)
- Session persistence across interactions via `--session-id`
- Context window compaction when conversations get long
- Streaming JSON output via `--output-format stream-json`
- Built-in retry with backoff on failures
- Hooks system (settings.json) for safety and routing
- Built-in `Read` and `Write` file tools for self-updating memory

**What it does NOT provide (we build these):**
- Voice pipeline (STT → router → TTS)
- Custom MCP tool servers (Home Assistant, TTS, weather, timers)
- CLAUDE.md content (personality, memory, entity list, behavioral rules)
- Stream filtering (route `speak` tool calls to TTS, suppress internal reasoning from voice)
- Redis pub/sub bridge to the rest of the pipeline
- Offline fallback (YAML routines only, no LLM)

**Accepted tradeoffs:**
- Requires Node.js 22+ on the Jetson (Claude Code CLI is Node-based)
- ~500ms–1s subprocess startup on first query per session
- Cloud-only for LLM — no local fallback via Ollama (YAML routines cover offline)
- Single provider (Anthropic) — no model switching
- Tools run as MCP servers (separate processes) rather than in-process

**Why Claude Code CLI over alternatives:**
- vs. Claude Agent SDK: Agent SDK requires separate API billing (API key only, no subscription auth). Claude Code CLI bills against Max subscription — no extra cost.
- vs. raw Claude API: saves building the entire tool-calling loop, session management, and context compaction from scratch. Also requires separate API billing.
- vs. OpenClaw: no separate Node.js gateway, simpler deployment, built-in session management
- vs. local LLM (Ollama): Claude is far more capable for tool use and reasoning. YAML routines cover offline.

**Billing:**
- Claude Code CLI authenticates via OAuth (`claude login`) on the Jetson
- All usage bills against the Claude Max subscription ($100–200/month flat)
- No API key or per-token charges needed
- Cost is predictable and bundled with the subscription

---

## 3. System Components

### 3.1 Voice Pipeline (Context — already designed)

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
                              Kokoro / Silero TTS
                                      │
                                      ▼
                                   Speaker
```

The LLM layer sits at the "Claude Code CLI" box. Everything else is handled by other services communicating via Redis pub/sub.

### 3.2 Claude Code CLI Integration

**Entry point:** A Python service (`jarvis_brain.py`) that:
1. Subscribes to Redis channel `stt_result`
2. Receives transcribed text
3. Calls `claude` CLI as a subprocess with `--session-id` and `--output-format stream-json`
4. Parses streaming JSON, routing `speak` tool calls to Redis channel `tts_request`
5. Logs all events to JSONL transcript

```python
# Simplified flow (not production code)
import subprocess
import json

async def query_claude(user_text: str, session_id: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", user_text,
        "--session-id", session_id,  # must be a UUID
        "--output-format", "stream-json",
        "--verbose",                 # required with stream-json in -p mode
        "--mcp-config", str(MCP_CONFIG_PATH),
        "--allowedTools", "mcp__jarvis__speak",  # Phase 1: lock to speak only
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(JARVIS_HOME),  # so Claude Code reads ~/.jarvis/CLAUDE.md
    )

    async for line in proc.stdout:
        event = json.loads(line)
        if is_speak_tool_call(event):
            await redis.publish("tts_request", extract_speech(event))
        log_to_jsonl(event)
```

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

**Session persistence:** Claude Code CLI requires `--session-id` to be a **UUID** (not an arbitrary string). Use `--session-id <uuid>` to *create* a new session, and `--resume <uuid>` to *continue* an existing one — reusing `--session-id` on an existing UUID errors with "Session ID already in use".

Strategy: derive one UUID per calendar day (e.g. `uuid.uuid5(NS, date.isoformat())`), try `--resume` first, fall back to `--session-id` on the first call of the day.

```python
import uuid, datetime as dt
JARVIS_NS = uuid.UUID("6f3c1b1e-7a2a-4e0f-9e7b-4a8b5c2d1e00")
session_uuid = str(uuid.uuid5(JARVIS_NS, dt.date.today().isoformat()))
# Day 1: claude -p ... --session-id <uuid>
# Day 1, turn 2+: claude -p ... --resume <uuid>
```

**Daily session rotation:** A new UUID is derived at midnight automatically. Before rotating, the brain service sends a final prompt asking Claude to summarize key facts and update CLAUDE.md's Memory section using the built-in Write tool.

**JSONL transcript:** Every message (user input, assistant response, tool calls, tool results) is appended to `~/.jarvis/logs/YYYY-MM-DD.jsonl`. One JSON object per line. Human-readable, greppable.

```json
{"ts": "2026-04-04T10:15:00Z", "type": "user", "text": "Turn on the lights"}
{"ts": "2026-04-04T10:15:01Z", "type": "tool_call", "tool": "ha_control", "args": {"entity_id": "light.living_room", "service": "turn_on"}}
{"ts": "2026-04-04T10:15:01Z", "type": "tool_result", "tool": "ha_control", "result": "Done"}
{"ts": "2026-04-04T10:15:02Z", "type": "tool_call", "tool": "speak", "args": {"text": "Living room lights are on, sir."}}
```

### 3.7 Stream Filtering

Claude Code CLI with `--output-format stream-json` emits one JSON object per line. Our stream filter parses these:

| Event type | Action |
|---|---|
| Assistant text | Log to dashboard + JSONL. Do NOT send to TTS. |
| Tool use (`speak`) | Extract text, publish to `tts_request` Redis channel |
| Tool use (`ha_control`, etc.) | Log to dashboard, tool executes via MCP server |
| Tool result | Log to JSONL |
| Message end | Update state machine to IDLE or SPEAKING |

Key insight: because `speak` is a tool, there is a clean separation. All spoken output goes through the tool. All other text is internal reasoning that gets logged but never vocalized.

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
| Claude Code CLI | LLM brain (subprocess) | `npm install -g @anthropic-ai/claude-code` |
| Node.js 22+ | Claude Code runtime | `apt install nodejs` or nvm |
| Python 3.10+ | All services | Pre-installed on JetPack |
| Redis | Event bus | `apt install redis-server` |
| `mcp` | MCP server framework | `pip install mcp` |
| `rapidfuzz` | Fuzzy match for command router | `pip install rapidfuzz` |
| `aiohttp` | Home Assistant API client | `pip install aiohttp` |
| `pyyaml` | Routine YAML parsing | `pip install pyyaml` |

### Authentication

```bash
# One-time setup: authenticate Claude Code with Max subscription
claude login
# This opens a browser OAuth flow — no API key needed
```

### Environment Variables

```bash
# No ANTHROPIC_API_KEY needed — Claude Code uses OAuth subscription auth
export HA_URL="http://192.168.1.x:8123"
export HA_TOKEN="eyJ..."
export JARVIS_HOME="$HOME/.jarvis"
```

---

## 7. Memory Budget (Updated)

```
OS + system services       ~0.7 GB
WhisperTRT small           ~1.0 GB
Kokoro + Silero TTS        ~0.2 GB  (or CosyVoice2-0.5B ~3–4 GB)
openWakeWord + Silero VAD  ~0.05 GB
Redis + orchestrator       ~0.1 GB
Node.js (Claude Code CLI)  ~0.1 GB
MCP server (Python)        ~0.05 GB
Python services            ~0.1 GB
───────────────────────────────────
Total                      ~2.3 GB  (with Kokoro/Silero)
                           ~5.2 GB  (with CosyVoice2)
Available                  8.0 GB
Headroom                   ~2.8–5.7 GB
```

Note: LLM runs in the cloud (Claude Max subscription), so no local LLM memory needed. GPU is free for WhisperTRT during STT, then idle during LLM processing, then available for CosyVoice TTS if using GPU mode.

---

## 8. Build Phases

### Phase 1 — Minimal voice loop (Week 1–2)
**Goal:** Speak to Jarvis, get a spoken response.

- [ ] Install Claude Code CLI + Node.js on Jetson *(dev machine done, Jetson pending)*
- [x] Run `claude login` to authenticate with Max subscription
- [x] Write `CLAUDE.md` with basic personality and voice rules
- [x] Implement MCP server with `speak` tool
- [x] Write `mcp_config.json` pointing to the MCP server
- [x] Wire: hardcoded text input → `claude -p` subprocess → `speak` tool → stdout *(TTS deferred to Phase 2)*
- [x] Verify the LLM only speaks via the `speak` tool, never outputs bare text
- [x] Add basic JSONL logging

**Phase 1 gotchas (learned in build):**
- `--session-id` must be a UUID. Use `--resume` for subsequent turns.
- `--output-format stream-json` requires `--verbose` in `-p` mode, otherwise claude exits silently.
- MCP server `command` must be an absolute path to the venv's `python`, not `python3`.
- Scope tools with `--allowedTools mcp__jarvis__speak` so Claude can't fall back to built-in tools.
- MCP tools are namespaced as `mcp__<server>__<tool>` in stream-json events — match by suffix.

**Milestone:** Type a sentence, hear Jarvis respond in the Jarvis voice.

### Phase 2 — Voice pipeline integration (Week 3–4)
**Goal:** Full wake-to-speak loop.

- [ ] Connect openWakeWord → Silero VAD → WhisperTRT (already designed)
- [ ] Implement Redis pub/sub bridge: `stt_result` → brain → `tts_request`
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

### Phase 4 — Memory and personality (Week 7–8)
**Goal:** Jarvis remembers and learns.

- [ ] Enable self-updating CLAUDE.md (Claude Code writes to Memory section via built-in Write tool)
- [ ] Implement daily session rotation with `--session-id jarvis-YYYY-MM-DD`
- [ ] Add `create_routine` tool (LLM creates YAML routines from conversation)
- [ ] Add proactive cron: morning briefing, scheduled reminders
- [ ] Tune CLAUDE.md personality through real usage
- [ ] Implement web dashboard (FastAPI + HTMX): conversation log, system status, config

**Milestone:** "Jarvis, remember I prefer 23 degrees" → next day, thermostat defaults to 23.

### Phase 5 — Polish and expansion (Week 9+)
**Goal:** Production quality.

- [ ] Add CosyVoice2 voice cloning (Jarvis voice for both EN and RU)
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
| Claude Max subscription rate limited | Slow or refused LLM responses | YAML routines still work. `speak` tool falls back to "I'm having trouble connecting." Monitor usage against Max limits. |
| Claude Code CLI subprocess latency | Slow first response | Keep session warm with `--session-id`. Pre-spawn if possible. Accept 500ms startup. |
| Node.js on Jetson ARM64 issues | CLI won't run | Node 22 LTS has official ARM64 builds. Test early in Phase 1. |
| CLAUDE.md grows too large | Context window overflow | Claude Code has built-in compaction. Daily session rotation trims history. |
| Prompt injection via HA sensor data | Unintended actions | Safety gate hook blocks dangerous actions. Sanitize all tool results. |
| TTS latency with CosyVoice | Slow voice output | Start with Kokoro (fast). CosyVoice is Phase 5. Pre-cache common phrases. |
| Max subscription plan changes | Pricing or feature changes | Monitor Anthropic announcements. API key fallback is easy to add later. |
| MCP server crashes | Tools unavailable | supervisord restarts MCP server. Claude Code handles tool errors gracefully. |

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
