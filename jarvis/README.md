# Jarvis — Phase 2

Full routing path:

```
stt_result ─► router ─► routine engine ─► tts_request
                └─► llm_request ─► brain (claude -p) ─► speak tool ─► tts_request
```

## Layout

```
jarvis/
├── CLAUDE.md                # personality + voice rules (auto-loaded by claude)
├── config.yaml              # redis url, fuzzy threshold, allowed tools
├── mcp_config.json          # points Claude Code at the MCP server
├── routines/
│   └── default.yaml         # built-in fuzzy-matched routines
├── services/
│   ├── jarvis_bus.py        # Redis channel names + pub/sub helpers
│   ├── jarvis_state.py      # State enum + StateMachine
│   ├── jarvis_router.py     # stt_result dispatcher, routines, barge-in
│   ├── jarvis_brain.py      # llm_request consumer, wraps claude -p
│   └── jarvis_mcp_server.py # `speak` tool -> tts_request
└── logs/
    └── YYYY-MM-DD.jsonl     # append-only transcript
```

## Prereqs

```bash
npm install -g @anthropic-ai/claude-code
claude login                  # OAuth with Claude Max subscription
pip install -r ../requirements.txt
redis-server &                # or brew services start redis
```

## Run (Phase 2 daemons)

Two long-running processes — in separate terminals or under supervisord:

```bash
cd jarvis
python services/jarvis_router.py          # subscribes to stt_result, wake_detected, tts_done
python services/jarvis_brain.py --daemon  # subscribes to llm_request
```

Then simulate STT input from a third terminal:

```bash
python services/jarvis_router.py --say "good morning jarvis"
python services/jarvis_router.py --say "what is the weather in almaty"
```

Watch the event bus:

```bash
redis-cli psubscribe 'tts_request' 'llm_request' 'state_change' 'barge_in'
```

## Phase 1 one-shot (still works)

```bash
python services/jarvis_brain.py "hello, introduce yourself briefly"
```

## How routing works

1. Router subscribes to `stt_result`.
2. For each utterance, fuzzy-match against triggers in `routines/*.yaml`:
   - Short single-word triggers (`stop`, `cancel`) require a **word-boundary** hit.
     (Otherwise `partial_ratio` would match "cancel" in "capital of france".)
   - Longer triggers use `rapidfuzz.partial_ratio` with threshold 80.
3. On match → execute the routine's actions in order:
   - `speak` → publishes `tts_request`
   - `ha_control` → Phase 3 stub (logs only for now)
   - `system: stop_tts` → publishes `barge_in`
   - `defer_to_llm` → publishes `llm_request` with a custom prompt
4. On miss → publish `llm_request` with the raw text; brain takes over.

## Barge-in

If a new `stt_result` or `wake_detected` arrives while state is `SPEAKING`,
the router publishes `barge_in` so the TTS service can cut audio mid-sentence.

## State machine

Owned by the router, published to `state_change`:

    IDLE → LISTENING → PROCESSING → SPEAKING → IDLE

The brain and TTS services observe state but do not own it.

## Next (Phase 3)

- Real `ha_control` / `ha_query` tools in the MCP server (aiohttp + HA REST)
- Safety gate hook in `.claude/settings.json`
- `set_timer`, `get_weather` tools
