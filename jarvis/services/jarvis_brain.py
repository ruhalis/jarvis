"""Brain — AsyncAnthropic SDK adapter for Jarvis Mode A.

Replaces the previous `claude -p` subprocess implementation. Now uses the
Anthropic Python SDK directly so we get:

  - explicit `cache_control: ephemeral` on system block + tool defs (A3)
  - per-turn token usage logged to JSONL incl. cache_read_input_tokens
  - cheap escalation from Haiku 4.5 → Sonnet 4.6 when Haiku omits speak (A5)
  - local per-day session files at sessions/YYYY-MM-DD.json (A4 redo)

The `speak` tool is implemented inline — it publishes to Redis `tts_request`,
the same shape the MCP server used. This drops the MCP subprocess from the
brain's tool path entirely; the MCP server stays available for other clients
(claude CLI, claw-code) but is no longer on this hot path.

Usage:

    # Daemon — subscribe to Redis llm_request:
    python jarvis_brain.py --daemon

    # One-shot (manual testing):
    python jarvis_brain.py "what's the weather like?"
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_bus as bus  # noqa: E402

DEBUG = bool(os.environ.get("JARVIS_DEBUG"))

JARVIS_HOME = Path(__file__).resolve().parent.parent
CLAUDE_MD = JARVIS_HOME / "CLAUDE.md"
LOG_DIR = JARVIS_HOME / "logs"
SESSION_DIR = JARVIS_HOME / "sessions"
STATE_DIR = JARVIS_HOME / "cache"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DEFAULT = "claude-haiku-4-5-20251001"
MODEL_ESCALATE = "claude-sonnet-4-6"
MAX_TOKENS = 1024
ESCALATION_DAILY_CAP = 20

SPEAK_TOOL = {
    "name": "speak",
    "description": (
        "Say something to the user via the speaker. This is the ONLY way to "
        "produce spoken output — every response to the user MUST be a speak "
        "call. Keep responses to 1-3 sentences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The exact words to speak aloud.",
            },
            "language": {
                "type": "string",
                "enum": ["en", "ru"],
                "description": "BCP-47 short code; default 'en'.",
            },
        },
        "required": ["text"],
    },
    "cache_control": {"type": "ephemeral"},
}


# ---------- logging / sessions ----------

def _log(event: dict) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _session_path(today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    return SESSION_DIR / f"{today.isoformat()}.json"


def _load_session() -> list[dict]:
    p = _session_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[brain] failed to load session ({exc}); starting fresh", file=sys.stderr)
        return []


def _save_session(messages: list[dict]) -> None:
    _session_path().write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_system() -> str:
    if CLAUDE_MD.exists():
        return CLAUDE_MD.read_text(encoding="utf-8")
    return "You are Jarvis. Always respond using the `speak` tool."


# Escalation cap (per-day, persisted)

def _escalation_state_path() -> Path:
    return STATE_DIR / "escalation_state.json"


def _escalations_today() -> int:
    p = _escalation_state_path()
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if data.get("date") != dt.date.today().isoformat():
        return 0
    return int(data.get("count", 0))


def _bump_escalations() -> int:
    count = _escalations_today() + 1
    _escalation_state_path().write_text(
        json.dumps({"date": dt.date.today().isoformat(), "count": count}),
        encoding="utf-8",
    )
    return count


# ---------- speak tool ----------

async def _handle_speak(redis_client, tool_input: dict) -> str:
    text = (tool_input.get("text") or "").strip()
    language = tool_input.get("language") or "en"
    _log({"type": "speak", "language": language, "text": text})
    if redis_client is not None and text:
        try:
            await bus.publish(
                redis_client,
                bus.CH_TTS_REQUEST,
                {"text": text, "lang": language, "priority": "normal"},
            )
        except Exception as exc:
            print(f"[brain] tts_request publish failed: {exc}", file=sys.stderr)
    print(f"<<SPEAK lang={language}>> {text}")
    return f"[Spoken in {language}]: {text}"


# ---------- core turn ----------

async def _run_turn(
    client: AsyncAnthropic,
    redis_client,
    user_text: str,
    model: str,
    system_text: str,
    history: list[dict],
) -> tuple[bool, list[dict]]:
    """Run one user turn. Returns (spoke, updated_history).

    `spoke` is True iff the model called `speak` at least once.
    """
    spoke = False
    messages = list(history)
    messages.append({"role": "user", "content": user_text})

    # Tool-use loop. Cap iterations defensively.
    for _ in range(6):
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[SPEAK_TOOL],
            messages=messages,
        ) as stream:
            response = await stream.get_final_message()

        usage = response.usage
        _log(
            {
                "type": "usage",
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "stop_reason": response.stop_reason,
            }
        )
        if DEBUG:
            print(
                f"[brain] {model} stop={response.stop_reason} "
                f"in={usage.input_tokens} out={usage.output_tokens} "
                f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}",
                file=sys.stderr,
            )

        # Append assistant response (preserving tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # End of turn — no more tools requested.
            break

        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            if block.name == "speak":
                spoke = True
                result = await _handle_speak(redis_client, block.input or {})
            else:
                result = f"Unknown tool: {block.name}"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )
        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason != "tool_use":
            break

    return spoke, messages


async def handle_user_turn(
    client: AsyncAnthropic,
    redis_client,
    user_text: str,
) -> None:
    system_text = _load_system()
    history = _load_session()
    _log({"type": "user", "text": user_text})

    spoke, new_history = await _run_turn(
        client, redis_client, user_text, MODEL_DEFAULT, system_text, history
    )

    if not spoke and _escalations_today() < ESCALATION_DAILY_CAP:
        count = _bump_escalations()
        _log({"type": "escalation", "to": MODEL_ESCALATE, "count_today": count})
        if DEBUG:
            print(f"[brain] escalating to {MODEL_ESCALATE} (#{count})", file=sys.stderr)
        # Re-run from the original history so we don't carry the failed turn.
        spoke, new_history = await _run_turn(
            client, redis_client, user_text, MODEL_ESCALATE, system_text, history
        )

    if not spoke:
        # Model failed to call speak even after escalation — emit a fallback
        # so the user isn't left in silence.
        if redis_client is not None:
            try:
                await bus.publish(
                    redis_client,
                    bus.CH_TTS_REQUEST,
                    {"text": "Apologies, sir — I lost my thread.", "lang": "en"},
                )
            except Exception:
                pass
        _log({"type": "no_speak", "user_text": user_text})

    _save_session(new_history)


# ---------- daemon / one-shot ----------

async def daemon() -> int:
    client = AsyncAnthropic()
    redis_client = bus.get_client()
    print("[brain] subscribed to llm_request (SDK mode)", file=sys.stderr)
    async for _chan, payload in bus.subscribe(redis_client, bus.CH_LLM_REQUEST):
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        try:
            await handle_user_turn(client, redis_client, text)
        except Exception as exc:
            _log({"type": "error", "where": "brain", "error": str(exc)})
            print(f"[brain] error: {exc}", file=sys.stderr)
    return 0


async def one_shot(user_text: str) -> int:
    client = AsyncAnthropic()
    try:
        redis_client = bus.get_client()
        await redis_client.ping()
    except Exception:
        redis_client = None
        print("[brain] redis unavailable; speak will log only", file=sys.stderr)
    await handle_user_turn(client, redis_client, user_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis brain (Anthropic SDK)")
    parser.add_argument("--daemon", action="store_true", help="subscribe to Redis llm_request")
    parser.add_argument("text", nargs="*", help="one-shot input text")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[brain] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    if args.daemon:
        return asyncio.run(daemon())
    if not args.text:
        parser.print_usage(sys.stderr)
        return 2
    return asyncio.run(one_shot(" ".join(args.text)))


if __name__ == "__main__":
    raise SystemExit(main())
