"""MCP server — exposes Jarvis custom tools.

Phase 2: `speak` publishes to Redis `tts_request` so the TTS service can
pick it up. Local JSONL logging is preserved for debugging/replay.

Redis publish is best-effort: if Redis is down we still log + return so the
LLM sees a successful tool result. (The TTS service is the source of truth
for whether audio actually played; it emits `tts_done` on success.)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import redis
from mcp.server.fastmcp import FastMCP

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
CH_TTS_REQUEST = "tts_request"

server = FastMCP("jarvis")

# Synchronous client: MCP tool handlers are sync here, and publish is cheap.
try:
    _redis_client: redis.Redis | None = redis.from_url(REDIS_URL, decode_responses=True)
    _redis_client.ping()
except Exception as exc:  # pragma: no cover — dev may run without redis
    print(f"[mcp] redis unavailable ({exc}); speak will log only", file=sys.stderr)
    _redis_client = None


def _log_spoken(text: str, language: str) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "type": "speak",
        "language": language,
        "text": text,
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@server.tool()
def speak(text: str, language: str = "en") -> str:
    """Say something to the user via the speaker.

    This is the ONLY way to produce spoken output. Use it for every response.

    Args:
        text: The exact words to speak aloud.
        language: BCP-47 short code — "en" or "ru".
    """
    _log_spoken(text, language)
    print(f"[SPEAK:{language}] {text}", file=sys.stderr, flush=True)
    if _redis_client is not None:
        try:
            _redis_client.publish(
                CH_TTS_REQUEST,
                json.dumps(
                    {"text": text, "lang": language, "priority": "normal"},
                    ensure_ascii=False,
                ),
            )
        except Exception as exc:  # keep tool result successful either way
            print(f"[mcp] redis publish failed: {exc}", file=sys.stderr)
    return f"[Spoken in {language}]: {text}"


if __name__ == "__main__":
    server.run()
