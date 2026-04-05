"""Brain — wraps `claude -p` and bridges Redis to Claude Code CLI.

Two modes:

    # Daemon (Phase 2): subscribe to Redis `llm_request`, run claude, let
    # the MCP server's `speak` tool publish to `tts_request`.
    python jarvis_brain.py --daemon

    # One-shot (Phase 1 behaviour, kept for manual testing):
    python jarvis_brain.py "what's the weather like?"

The brain itself does NOT publish to `tts_request` — that happens inside
the MCP server when the model calls the `speak` tool. The brain's job is
to: receive a request, spawn claude, parse stream-json events, log them,
and manage state (PROCESSING/SPEAKING/IDLE).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_bus as bus  # noqa: E402

DEBUG = bool(os.environ.get("JARVIS_DEBUG"))

JARVIS_HOME = Path(__file__).resolve().parent.parent
MCP_CONFIG = JARVIS_HOME / "mcp_config.json"
LOG_DIR = JARVIS_HOME / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Use a stable namespace so daily UUIDs are reproducible across restarts.
JARVIS_NS = uuid.UUID("6f3c1b1e-7a2a-4e0f-9e7b-4a8b5c2d1e00")


def daily_session_uuid(today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    return str(uuid.uuid5(JARVIS_NS, today.isoformat()))


def _log_event(event: dict) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_speak_calls(event: dict) -> list[dict]:
    calls: list[dict] = []
    if event.get("type") != "assistant":
        return calls
    message = event.get("message") or {}
    for block in message.get("content", []) or []:
        if block.get("type") != "tool_use":
            continue
        # MCP tools are namespaced as "mcp__<server>__<tool>"
        if block.get("name", "").endswith("speak"):
            calls.append(block.get("input") or {})
    return calls


# Tracks sessions we've created this process, so we know whether to use
# --session-id (first turn) or --resume (subsequent turns).
_seen_sessions: set[str] = set()


async def _run_claude(user_text: str, session_id: str) -> int:
    """Spawn claude, stream events, log them. Return process exit code."""
    args = [
        "claude",
        "-p",
        user_text,
        "--output-format",
        "stream-json",
        "--verbose",
        "--mcp-config",
        str(MCP_CONFIG),
        "--allowedTools",
        "mcp__jarvis__speak",
    ]
    # --resume for an existing UUID, --session-id to create it.
    if session_id in _seen_sessions:
        args += ["--resume", session_id]
    else:
        args += ["--session-id", session_id]

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(JARVIS_HOME),
    )
    assert proc.stdout is not None

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"[non-json] {line}", file=sys.stderr)
            continue

        if DEBUG:
            print(
                f"[event] {event.get('type')}: {json.dumps(event)[:300]}",
                file=sys.stderr,
            )

        # Log every event for replay / debugging.
        _log_event({"type": "claude_event", "event_type": event.get("type")})

        for call in _extract_speak_calls(event):
            text = call.get("text", "")
            lang = call.get("language", "en")
            # Speak publishing happens inside the MCP server; here we just log.
            _log_event({"type": "tool_call", "tool": "speak", "text": text, "lang": lang})
            # Also echo for manual dev runs.
            print(f"<<SPEAK lang={lang}>> {text}")

    rc = await proc.wait()
    err = (await proc.stderr.read()).decode("utf-8", errors="replace")
    if rc != 0 or DEBUG:
        if err:
            print(f"[claude stderr rc={rc}]\n{err}", file=sys.stderr)
    if rc == 0:
        _seen_sessions.add(session_id)
    return rc


# -------- daemon mode --------

async def daemon() -> int:
    client = bus.get_client()
    print("[brain] subscribed to llm_request", file=sys.stderr)
    async for _chan, payload in bus.subscribe(client, bus.CH_LLM_REQUEST):
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        _log_event({"type": "user", "text": text})
        try:
            session_id = daily_session_uuid()
            await _run_claude(text, session_id)
        except Exception as exc:
            _log_event({"type": "error", "where": "brain", "error": str(exc)})
            print(f"[brain] error: {exc}", file=sys.stderr)
    return 0


# -------- one-shot mode (kept for manual testing) --------

async def one_shot(user_text: str) -> int:
    session_id = daily_session_uuid()
    return await _run_claude(user_text, session_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis brain")
    parser.add_argument("--daemon", action="store_true", help="subscribe to Redis llm_request")
    parser.add_argument("text", nargs="*", help="one-shot input text")
    args = parser.parse_args()

    if args.daemon:
        return asyncio.run(daemon())
    if not args.text:
        parser.print_usage(sys.stderr)
        return 2
    return asyncio.run(one_shot(" ".join(args.text)))


if __name__ == "__main__":
    raise SystemExit(main())
