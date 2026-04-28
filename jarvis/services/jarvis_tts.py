"""TTS service — consumes `tts_request`, plays audio, publishes levels.

Subscribes:
    tts_request   {text, lang, priority}
    barge_in      cancel current playback

Publishes:
    tts_state     {engine: "kokoro"|"say"|"espeak"|"null"}
    tts_level     {rms: 0.0-1.0}      ~30 Hz while speaking
    tts_done      {ok: bool, reason?: str}

Engine selection (config.yaml `tts.engine`):
    "auto"   — kokoro if importable, else `say` on darwin, `espeak` on linux, else `null`
    "kokoro" — Jetson production path (lazy-imported)
    "say"    — macOS dev fallback (subprocess `say`, no levels, no barge-in)
    "espeak" — Linux dev fallback (subprocess `espeak`, no levels, no barge-in)
    "null"   — silent: log only. Useful for headless dev without a speaker.

The cache layer keys on (engine, voice, lang, text) and stores WAVs under
`cache/tts/`. Cache hits skip synthesis entirely.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_bus as bus  # noqa: E402

log = logging.getLogger("jarvis.tts")

JARVIS_HOME = Path(__file__).resolve().parent.parent
CONFIG_PATH = JARVIS_HOME / "config.yaml"
LOG_DIR = JARVIS_HOME / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- config ----------


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


# ---------- engines ----------


class Engine:
    name: str = "null"

    async def synth_to_wav(self, text: str, lang: str, out_path: Path) -> bool:
        """Render `text` to a 16-bit PCM WAV at `out_path`. Return True on success."""
        return False

    async def speak_direct(self, text: str, lang: str) -> int:
        """Engines that can't produce WAVs (e.g. system `say`) speak directly.
        Return process-style exit code: 0 on success.
        """
        return 0

    @property
    def produces_wav(self) -> bool:
        return False


class NullEngine(Engine):
    name = "null"

    async def speak_direct(self, text: str, lang: str) -> int:
        log.info("[null engine] would speak: %s", text)
        return 0


class SayEngine(Engine):
    """macOS `say` — speaks directly, no WAV/RMS, no mid-utterance barge-in
    (we can still kill the process)."""

    name = "say"

    async def speak_direct(self, text: str, lang: str) -> int:
        # Voice mapping: rough defaults; user can change in config later.
        voice = "Daniel" if lang == "en" else None
        args = ["say"]
        if voice:
            args += ["-v", voice]
        args.append(text)
        proc = await asyncio.create_subprocess_exec(*args)
        return await proc.wait()


class EspeakEngine(Engine):
    name = "espeak"

    async def speak_direct(self, text: str, lang: str) -> int:
        args = ["espeak", "-v", lang, text]
        proc = await asyncio.create_subprocess_exec(*args)
        return await proc.wait()


class KokoroEngine(Engine):
    """Lazy-imported Kokoro engine. Produces 24 kHz mono float32 audio,
    written as 16-bit PCM WAV.

    On the Jetson this is the default. We don't import at module import
    time so dev machines without `kokoro` installed can still run the
    service with `engine: say` or `engine: null`.
    """

    name = "kokoro"

    def __init__(self, voice: str, speed: float) -> None:
        from kokoro import KPipeline  # type: ignore

        # "a" = American English. Kokoro auto-detects per-voice; we keep
        # a single pipeline since we only do en for now.
        self._pipe = KPipeline(lang_code="a")
        self._voice = voice
        self._speed = speed

    @property
    def produces_wav(self) -> bool:
        return True

    async def synth_to_wav(self, text: str, lang: str, out_path: Path) -> bool:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        def _run() -> bool:
            chunks: list[Any] = []
            for _gs, _ps, audio in self._pipe(text, voice=self._voice, speed=self._speed):
                chunks.append(audio)
            if not chunks:
                return False
            full = np.concatenate(chunks).astype("float32")
            sf.write(str(out_path), full, 24000, subtype="PCM_16")
            return True

        return await asyncio.to_thread(_run)


def _select_engine(cfg: dict) -> Engine:
    name = (cfg.get("tts", {}).get("engine") or "auto").lower()
    voice = cfg.get("tts", {}).get("kokoro_voice") or "af_heart"
    speed = float(cfg.get("tts", {}).get("kokoro_speed") or 1.0)

    def try_kokoro() -> Engine | None:
        try:
            return KokoroEngine(voice=voice, speed=speed)
        except Exception as exc:  # noqa: BLE001
            log.warning("kokoro unavailable (%s)", exc)
            return None

    if name == "kokoro":
        eng = try_kokoro()
        if eng is None:
            log.warning("falling back to null engine")
            return NullEngine()
        return eng
    if name == "say":
        return SayEngine()
    if name == "espeak":
        return EspeakEngine()
    if name == "null":
        return NullEngine()

    # auto
    eng = try_kokoro()
    if eng is not None:
        return eng
    if platform.system() == "Darwin":
        return SayEngine()
    if platform.system() == "Linux":
        return EspeakEngine()
    return NullEngine()


# ---------- cache ----------


def _cache_key(engine_name: str, voice: str | None, lang: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(f"{engine_name}|{voice or ''}|{lang}|{text}".encode("utf-8"))
    return h.hexdigest()[:16]


# ---------- playback ----------


async def _play_wav(
    path: Path,
    *,
    cancel: asyncio.Event,
    publish_level: callable,  # type: ignore[valid-type]
    level_hz: int,
) -> bool:
    """Play `path` with cancellation + RMS publishing. Returns False if cancelled."""
    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("sounddevice/soundfile unavailable (%s); using fallback player", exc)
        return await _play_wav_fallback(path, cancel=cancel)

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)

    chunk_samples = max(1, sr // max(1, level_hz))
    cursor = 0

    loop = asyncio.get_running_loop()
    done_evt = asyncio.Event()

    def _cb(outdata, frames, _time, status):  # noqa: ANN001
        nonlocal cursor
        if status:
            log.debug("sd status: %s", status)
        if cancel.is_set():
            outdata[:] = 0
            raise sd.CallbackStop
        end = cursor + frames
        chunk = data[cursor:end]
        if len(chunk) < frames:
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk) :, 0] = 0
            cursor = end
            loop.call_soon_threadsafe(done_evt.set)
            raise sd.CallbackStop
        outdata[:, 0] = chunk
        cursor = end
        # RMS of this chunk → 0..1
        rms = float((chunk.astype("float64") ** 2).mean() ** 0.5)
        loop.call_soon_threadsafe(publish_level, min(1.0, rms))

    stream = sd.OutputStream(
        samplerate=sr, channels=1, dtype="float32",
        blocksize=chunk_samples, callback=_cb,
    )
    stream.start()
    try:
        # Wait for either natural completion or barge-in cancel.
        done_task = asyncio.create_task(done_evt.wait())
        cancel_task = asyncio.create_task(cancel.wait())
        try:
            await asyncio.wait(
                {done_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (done_task, cancel_task):
                if not t.done():
                    t.cancel()
    finally:
        stream.stop()
        stream.close()
    return not cancel.is_set()


async def _play_wav_fallback(path: Path, *, cancel: asyncio.Event) -> bool:
    """No sounddevice — use system player. Loses RMS but keeps barge-in via SIGTERM."""
    if platform.system() == "Darwin":
        args = ["afplay", str(path)]
    else:
        args = ["aplay", "-q", str(path)]
    proc = await asyncio.create_subprocess_exec(*args)

    async def _wait_cancel() -> None:
        await cancel.wait()
        if proc.returncode is None:
            proc.terminate()

    canceller = asyncio.create_task(_wait_cancel())
    try:
        rc = await proc.wait()
    finally:
        canceller.cancel()
    return rc == 0 and not cancel.is_set()


# ---------- service ----------


def _log_event(event: dict) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TTSService:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._engine: Engine = _select_engine(cfg)
        self._voice = cfg.get("tts", {}).get("kokoro_voice")
        self._cache_dir = JARVIS_HOME / (cfg.get("tts", {}).get("cache_dir") or "cache/tts")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._publish_levels = bool(cfg.get("tts", {}).get("publish_levels", True))
        self._level_hz = int(cfg.get("tts", {}).get("level_hz") or 30)
        self._client = bus.get_client()
        self._cancel = asyncio.Event()
        self._current: asyncio.Task | None = None
        log.info("tts engine: %s", self._engine.name)

    # ----- helpers -----

    async def _safe_publish(self, channel: str, payload: dict) -> None:
        try:
            await bus.publish(self._client, channel, payload)
        except Exception as exc:  # noqa: BLE001 — Redis down is non-fatal for dev
            log.debug("publish to %s failed: %s", channel, exc)

    async def _publish_state(self) -> None:
        await self._safe_publish(bus.CH_TTS_STATE, {"engine": self._engine.name})

    def _publish_level(self, rms: float) -> None:
        if not self._publish_levels:
            return
        asyncio.create_task(self._safe_publish(bus.CH_TTS_LEVEL, {"rms": rms}))

    async def _publish_done(self, ok: bool, reason: str | None = None) -> None:
        payload: dict[str, Any] = {"ok": ok}
        if reason:
            payload["reason"] = reason
        await self._safe_publish(bus.CH_TTS_DONE, payload)

    # ----- one utterance -----

    async def _speak_one(self, text: str, lang: str) -> None:
        if not text.strip():
            await self._publish_done(False, "empty")
            return

        self._cancel.clear()
        _log_event({"type": "tts_start", "engine": self._engine.name, "lang": lang, "text": text})

        try:
            if self._engine.produces_wav:
                key = _cache_key(self._engine.name, self._voice, lang, text)
                wav_path = self._cache_dir / f"{key}.wav"
                if not wav_path.exists():
                    ok = await self._engine.synth_to_wav(text, lang, wav_path)
                    if not ok:
                        await self._publish_done(False, "synth_failed")
                        return
                played = await _play_wav(
                    wav_path,
                    cancel=self._cancel,
                    publish_level=self._publish_level,
                    level_hz=self._level_hz,
                )
                await self._publish_done(played, None if played else "barge_in")
            else:
                rc = await self._engine.speak_direct(text, lang)
                # speak_direct doesn't observe cancel mid-utterance for `say`;
                # a barge-in arriving here just lets the current word finish.
                ok = (rc == 0) and not self._cancel.is_set()
                await self._publish_done(ok, None if ok else "barge_in_or_error")
        except Exception as exc:  # noqa: BLE001
            log.exception("tts failed")
            _log_event({"type": "tts_error", "error": str(exc)})
            await self._publish_done(False, f"exception:{exc}")

    # ----- main loop -----

    async def run(self) -> int:
        await self._publish_state()
        log.info("[tts] subscribed to %s + %s", bus.CH_TTS_REQUEST, bus.CH_BARGE_IN)
        async for chan, payload in bus.subscribe(
            self._client, bus.CH_TTS_REQUEST, bus.CH_BARGE_IN,
        ):
            if chan == bus.CH_BARGE_IN:
                if self._current and not self._current.done():
                    log.info("[tts] barge-in — cancelling current utterance")
                    self._cancel.set()
                continue

            text = (payload.get("text") or "").strip()
            lang = payload.get("lang") or self._cfg.get("tts", {}).get("default_language") or "en"
            if not text:
                continue

            # If something is already speaking, cancel it before starting next.
            if self._current and not self._current.done():
                self._cancel.set()
                try:
                    await self._current
                except asyncio.CancelledError:
                    pass

            self._current = asyncio.create_task(self._speak_one(text, lang))
        return 0


# ---------- entrypoint ----------


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis TTS service")
    parser.add_argument("--text", help="one-shot: speak this and exit")
    parser.add_argument("--lang", default="en")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose or os.environ.get("JARVIS_DEBUG") else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = _load_config()

    if args.text:
        # One-shot mode for sanity checking the engine without Redis traffic.
        async def _once() -> int:
            svc = TTSService(cfg)
            await svc._speak_one(args.text, args.lang)  # noqa: SLF001
            return 0

        return asyncio.run(_once())

    svc = TTSService(cfg)
    return asyncio.run(svc.run())


if __name__ == "__main__":
    raise SystemExit(main())
