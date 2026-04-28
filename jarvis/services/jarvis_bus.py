"""Redis channel names and a tiny async pub/sub helper.

One place for channel strings so services don't drift. Everything routes
through Redis so the brain, router, STT, and TTS can live in separate
processes (supervisord-managed on the Jetson).
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import redis.asyncio as redis


# Channel names — keep these in sync with any new services.
CH_WAKE = "wake_detected"
CH_STT = "stt_result"
CH_STT_PARTIAL = "stt_partial"
CH_LLM_REQUEST = "llm_request"
CH_TTS_REQUEST = "tts_request"
CH_TTS_DONE = "tts_done"
CH_STATE = "state_change"
CH_BARGE_IN = "barge_in"
CH_TIMER_SET = "timer_set"
CH_TTS_LEVEL = "tts_level"
CH_TTS_STATE = "tts_state"


def redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")


def get_client() -> redis.Redis:
    return redis.from_url(redis_url(), decode_responses=True)


async def publish(client: redis.Redis, channel: str, payload: dict[str, Any]) -> None:
    await client.publish(channel, json.dumps(payload, ensure_ascii=False))


async def subscribe(client: redis.Redis, *channels: str) -> AsyncIterator[tuple[str, dict]]:
    """Async generator yielding (channel, decoded-json-payload) tuples."""
    pubsub = client.pubsub()
    await pubsub.subscribe(*channels)
    try:
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            chan = msg.get("channel")
            data = msg.get("data")
            try:
                payload = json.loads(data) if data else {}
            except json.JSONDecodeError:
                payload = {"_raw": data}
            yield chan, payload
    finally:
        await pubsub.unsubscribe(*channels)
        await pubsub.aclose()
