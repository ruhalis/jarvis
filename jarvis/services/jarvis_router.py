"""Command router + routine engine.

Subscribes to:
    wake_detected  — user started speaking (for barge-in)
    stt_result     — transcribed user utterance
    tts_done       — TTS finished speaking (return to IDLE)

Flow per utterance:
    1. If state == SPEAKING, publish barge_in and drop current TTS.
    2. Fuzzy-match utterance text against loaded routines.
       - score >= threshold: execute routine actions in order.
       - score <  threshold: publish llm_request -> brain.
    3. Routine actions:
        speak:         publish to tts_request
        ha_control:    (stub for Phase 2) log + placeholder publish
        system:        handle stop_tts etc.
        defer_to_llm:  publish llm_request with prompt

No LLM round trip on a match — that's the whole point of the fast path.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_bus as bus  # noqa: E402
from jarvis_state import State, StateMachine  # noqa: E402

log = logging.getLogger("jarvis.router")

JARVIS_HOME = Path(__file__).resolve().parent.parent
CONFIG_PATH = JARVIS_HOME / "config.yaml"
LOG_DIR = JARVIS_HOME / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- routine loading ----------

@dataclass
class Routine:
    name: str
    triggers: list[str]
    actions: list[dict]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_routines(paths: list[Path]) -> list[Routine]:
    out: list[Routine] = []
    for p in paths:
        data = _load_yaml(p)
        for i, item in enumerate(data.get("routines") or []):
            triggers = item.get("trigger") or []
            if isinstance(triggers, str):
                triggers = [triggers]
            out.append(
                Routine(
                    name=item.get("name") or f"{p.stem}_{i}",
                    triggers=[t.lower() for t in triggers],
                    actions=item.get("actions") or [],
                )
            )
    log.info("loaded %d routines", len(out))
    return out


def _word_bounded(text: str, trigger: str) -> bool:
    """True if trigger appears in text at word boundaries (Unicode-aware)."""
    pattern = r"(?:^|\W)" + re.escape(trigger) + r"(?:$|\W)"
    return re.search(pattern, text, flags=re.UNICODE) is not None


def match_routine(
    text: str, routines: list[Routine], threshold: int
) -> tuple[Routine, int] | None:
    """Return the best-scoring routine above threshold, else None.

    Short triggers (single word, <=8 chars like "stop", "cancel") need
    word-boundary matching — partial_ratio will otherwise match the
    trigger against any substring (e.g. "cancel" in "capital of france").
    Longer triggers fall through to fuzz.partial_ratio.
    """
    if not text:
        return None
    norm = text.lower().strip()
    best: tuple[Routine, int] | None = None
    for r in routines:
        for trig in r.triggers:
            if len(trig.split()) == 1 and len(trig) <= 8:
                score = 100 if _word_bounded(norm, trig) else 0
            else:
                score = int(fuzz.partial_ratio(trig, norm))
            if score >= threshold and (best is None or score > best[1]):
                best = (r, score)
    return best


# ---------- action execution ----------

def _log(entry: dict) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **entry}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Router:
    def __init__(self, client, routines: list[Routine], threshold: int) -> None:
        self._client = client
        self._routines = routines
        self._threshold = threshold
        self._sm = StateMachine(client)
        # Set True while we have an in-flight llm_request/tts; barge-in clears.
        self._awaiting_tts = False

    # --- dispatch ---

    async def on_stt(self, text: str, lang: str) -> None:
        _log({"type": "user", "text": text, "lang": lang})

        # Barge-in: if we're speaking, cut the current TTS and start fresh.
        if self._sm.state is State.SPEAKING:
            await self._barge_in()

        await self._sm.transition(State.PROCESSING)
        hit = match_routine(text, self._routines, self._threshold)
        if hit:
            routine, score = hit
            _log({"type": "router_match", "routine": routine.name, "score": score})
            await self._execute_actions(routine.actions, user_text=text)
        else:
            _log({"type": "router_miss", "text": text})
            await self._defer_to_llm(text)
        # After dispatching, we return to IDLE. TTS service will transition
        # to SPEAKING when it actually plays audio (via tts_request consumer).

    async def on_wake(self) -> None:
        if self._sm.state is State.SPEAKING:
            await self._barge_in()
        await self._sm.transition(State.LISTENING)

    async def on_tts_done(self) -> None:
        self._awaiting_tts = False
        if self._sm.state is State.SPEAKING:
            await self._sm.transition(State.IDLE)

    async def on_tts_request(self, _payload: dict) -> None:
        # Observer only: whenever anyone queues audio, move to SPEAKING.
        self._awaiting_tts = True
        if self._sm.state is not State.SPEAKING:
            await self._sm.transition(State.SPEAKING, force=True)

    # --- internals ---

    async def _barge_in(self) -> None:
        _log({"type": "barge_in"})
        await bus.publish(self._client, bus.CH_BARGE_IN, {"reason": "user_spoke"})
        self._awaiting_tts = False

    async def _defer_to_llm(self, text: str, note: str = "") -> None:
        payload = {"text": text}
        if note:
            payload["note"] = note
        await bus.publish(self._client, bus.CH_LLM_REQUEST, payload)

    async def _execute_actions(self, actions: list[dict], user_text: str) -> None:
        for action in actions:
            if not isinstance(action, dict) or len(action) != 1:
                log.warning("skipping malformed action: %r", action)
                continue
            ((kind, args),) = action.items()
            args = args or {}
            try:
                await self._run_action(kind, args, user_text)
            except Exception as exc:
                _log({"type": "action_error", "kind": kind, "error": str(exc)})
                log.exception("action %s failed", kind)

    async def _run_action(self, kind: str, args: dict, user_text: str) -> None:
        if kind == "speak":
            text = args.get("text") or ""
            lang = args.get("language") or "en"
            if not text:
                return
            await bus.publish(
                self._client,
                bus.CH_TTS_REQUEST,
                {"text": text, "lang": lang, "priority": "normal"},
            )
            _log({"type": "tool_call", "tool": "speak", "text": text, "lang": lang})
        elif kind == "ha_control":
            # Phase 3 will wire this to aiohttp + HA. For now just log so
            # we can see routines firing end-to-end.
            _log({"type": "tool_call", "tool": "ha_control", "args": args})
            log.info("[stub] ha_control %s", args)
        elif kind == "system":
            cmd = args.get("command")
            if cmd == "stop_tts":
                await bus.publish(self._client, bus.CH_BARGE_IN, {"reason": "stop_cmd"})
                _log({"type": "system", "command": "stop_tts"})
            else:
                log.warning("unknown system command: %r", cmd)
        elif kind == "defer_to_llm":
            prompt = args.get("prompt") or user_text
            _log({"type": "defer_to_llm", "prompt": prompt})
            await self._defer_to_llm(prompt, note="from_routine")
        else:
            log.warning("unknown action kind: %s", kind)


# ---------- main loop ----------

async def run(config: dict) -> int:
    client = bus.get_client()

    routines_paths = [JARVIS_HOME / p for p in config["router"]["routines_paths"]]
    routines = load_routines(routines_paths)
    threshold = int(config["router"]["fuzzy_threshold"])
    router = Router(client, routines, threshold)

    channels = (bus.CH_STT, bus.CH_WAKE, bus.CH_TTS_DONE, bus.CH_TTS_REQUEST)
    print(f"[router] subscribed to {channels}", file=sys.stderr)

    async for chan, payload in bus.subscribe(client, *channels):
        try:
            if chan == bus.CH_STT:
                await router.on_stt(
                    (payload.get("text") or "").strip(),
                    payload.get("lang") or "en",
                )
            elif chan == bus.CH_WAKE:
                await router.on_wake()
            elif chan == bus.CH_TTS_DONE:
                await router.on_tts_done()
            elif chan == bus.CH_TTS_REQUEST:
                await router.on_tts_request(payload)
        except Exception as exc:
            log.exception("handler error on %s: %s", chan, exc)
    return 0


def load_config() -> dict:
    cfg = _load_yaml(CONFIG_PATH)
    if os.environ.get("REDIS_URL"):
        cfg["redis_url"] = os.environ["REDIS_URL"]
    return cfg


def main() -> int:
    logging.basicConfig(
        level=logging.INFO if not os.environ.get("JARVIS_DEBUG") else logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Jarvis command router")
    parser.add_argument("--say", help="simulate an stt_result and exit")
    args = parser.parse_args()

    if args.say:
        # Quick manual test: publish to stt_result and exit.
        async def _once() -> int:
            client = bus.get_client()
            await bus.publish(client, bus.CH_STT, {"text": args.say, "lang": "en"})
            await client.aclose()
            print(f"published stt_result: {args.say!r}")
            return 0

        return asyncio.run(_once())

    return asyncio.run(run(load_config()))


if __name__ == "__main__":
    raise SystemExit(main())
