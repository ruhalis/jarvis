"""Voice capture pipeline — wake word + VAD + streaming STT.

Subscribes:
    barge_in           (informational; the mic loop never stops)

Publishes:
    wake_detected      {model: str, score: float}
    stt_partial        {text: str, stable: bool}
    stt_result         {text: str, lang: str}

Backends (config: `voice.mode`):
    "mic"   — real pipeline: openWakeWord -> Silero VAD -> Parakeet streaming.
              Lazy-imports the heavy deps so dev machines don't pay for them.
    "stdin" — read one utterance per line from stdin, publish wake_detected +
              stt_result. Lets us drive router/brain/tts end-to-end on the Mac
              without a microphone or NeMo build.
    "null"  — sit idle. For environments that just need the supervisor unit.
    "auto"  — try mic; on import error fall back to stdin.

The mic backend is intentionally a thin orchestrator. The STT engine is
encapsulated behind a small interface (`SttEngine`) so we can swap the
Parakeet implementation for a TensorRT export later (B2) without touching
the surrounding loop.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Protocol

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_bus as bus  # noqa: E402

log = logging.getLogger("jarvis.voice")

JARVIS_HOME = Path(__file__).resolve().parent.parent
CONFIG_PATH = JARVIS_HOME / "config.yaml"
LOG_DIR = JARVIS_HOME / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- config ----------


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _log_event(event: dict) -> None:
    log_file = LOG_DIR / f"{dt.date.today().isoformat()}.jsonl"
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------- STT engine interface ----------


class SttEngine(Protocol):
    """Streaming STT. `transcribe` consumes a single utterance worth of
    16 kHz mono int16 frames from `frames` and yields partial strings,
    finishing with the final transcript."""

    name: str

    async def transcribe(
        self, frames: AsyncIterator["bytes"]
    ) -> AsyncIterator[tuple[str, bool]]:
        """Yield (text, is_final). At least one tuple, last one is_final=True."""
        ...


# ---------- backends ----------


class StdinBackend:
    """Read one utterance per line from stdin and publish events.

    Useful on dev machines: `echo "what time is it" | python jarvis_voice.py`
    or run interactively. Each line publishes wake_detected then stt_result so
    the router exercises both barge-in and dispatch paths.
    """

    name = "stdin"

    def __init__(self, client, lang: str = "en") -> None:
        self._client = client
        self._lang = lang

    async def run(self) -> None:
        log.info("[voice/stdin] type an utterance and press enter (Ctrl-D to quit)")
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                log.info("[voice/stdin] EOF — exiting")
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            await bus.publish(
                self._client, bus.CH_WAKE, {"model": "stdin", "score": 1.0}
            )
            _log_event({"type": "wake", "source": "stdin"})
            await bus.publish(
                self._client, bus.CH_STT, {"text": text, "lang": self._lang}
            )
            _log_event({"type": "stt_result", "text": text, "lang": self._lang})


class NullBackend:
    name = "null"

    def __init__(self, client) -> None:
        self._client = client

    async def run(self) -> None:
        log.info("[voice/null] idle")
        await asyncio.Event().wait()


class MicBackend:
    """Real pipeline: openWakeWord -> Silero VAD -> streaming STT.

    Lazy-imports everything in `__init__` so the module stays importable on
    machines without NeMo/openWakeWord/silero. Construction failure is the
    signal for `auto` mode to fall back to stdin.
    """

    name = "mic"

    def __init__(self, client, cfg: dict) -> None:
        self._client = client
        v = cfg.get("voice") or {}
        self._sr = int(v.get("sample_rate") or 16000)
        self._frame_ms = int(v.get("frame_ms") or 20)
        self._frame_samples = self._sr * self._frame_ms // 1000
        self._wake_threshold = float(v.get("wake_threshold") or 0.5)
        self._wake_models = list(v.get("wake_models") or ["jarvis"])
        self._vad_min_silence_ms = int(v.get("vad_min_silence_ms") or 500)
        self._vad_speech_pad_ms = int(v.get("vad_speech_pad_ms") or 200)
        self._input_device = v.get("input_device")
        self._lang = "en"
        self._prewarm = bool(v.get("prewarm_partials"))

        # Lazy imports — any ImportError here surfaces to caller.
        import sounddevice as sd  # type: ignore  # noqa: F401
        from openwakeword.model import Model as OwwModel  # type: ignore
        import torch  # type: ignore  # noqa: F401

        self._oww = OwwModel(wakeword_models=self._wake_models)
        self._vad = self._load_silero()
        self._stt = self._load_parakeet(v.get("parakeet_model") or "nvidia/parakeet-tdt-0.6b-v3")

    @staticmethod
    def _load_silero():
        # Silero ships its own torch.hub entrypoint.
        import torch  # type: ignore

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        return {"model": model, "utils": utils}

    @staticmethod
    def _load_parakeet(model_name: str) -> "SttEngine":
        # Defer to a separate engine class so the swap to a TRT export later
        # (B2) is local.
        return ParakeetEngine(model_name)

    # --- main loop ---

    async def run(self) -> None:
        import sounddevice as sd  # type: ignore
        import numpy as np  # type: ignore

        loop = asyncio.get_running_loop()
        frame_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

        def _cb(indata, frames, _time, status):  # noqa: ANN001
            if status:
                log.debug("sd input status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            pcm = (mono * 32767).astype("int16").tobytes()
            try:
                loop.call_soon_threadsafe(frame_q.put_nowait, pcm)
            except asyncio.QueueFull:
                # Drop frame rather than block the audio thread.
                pass

        stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=self._frame_samples,
            device=self._input_device,
            callback=_cb,
        )
        stream.start()
        log.info(
            "[voice/mic] listening (sr=%d, frame=%d ms, wake=%s)",
            self._sr, self._frame_ms, ",".join(self._wake_models),
        )

        try:
            while True:
                await self._wait_for_wake(frame_q)
                await self._capture_and_transcribe(frame_q)
        finally:
            stream.stop()
            stream.close()

    async def _wait_for_wake(self, frame_q: asyncio.Queue) -> None:
        import numpy as np  # type: ignore

        while True:
            pcm = await frame_q.get()
            arr = np.frombuffer(pcm, dtype="int16")
            scores = self._oww.predict(arr)
            for name, score in scores.items():
                if score >= self._wake_threshold:
                    log.info("[voice/mic] wake: %s (%.2f)", name, score)
                    await bus.publish(
                        self._client, bus.CH_WAKE,
                        {"model": name, "score": float(score)},
                    )
                    _log_event({"type": "wake", "model": name, "score": float(score)})
                    return

    async def _capture_and_transcribe(self, frame_q: asyncio.Queue) -> None:
        """After wake, gate frames through Silero VAD until we hit
        `vad_min_silence_ms` of trailing silence, streaming partials as we
        go and publishing the final on endpoint."""
        # The VAD-gated frame iterator and the STT consumer run concurrently:
        # frames flow into both Silero (to decide endpoint) and the STT engine
        # (to produce text). We drive partials as they arrive.
        utterance_q: asyncio.Queue[bytes | None] = asyncio.Queue()
        vad_task = asyncio.create_task(
            self._vad_gate(frame_q, utterance_q)
        )

        async def _gen() -> AsyncIterator[bytes]:
            while True:
                item = await utterance_q.get()
                if item is None:
                    return
                yield item

        last_partial = ""
        try:
            async for text, is_final in self._stt.transcribe(_gen()):
                text = (text or "").strip()
                if not text:
                    continue
                if is_final:
                    await bus.publish(
                        self._client, bus.CH_STT,
                        {"text": text, "lang": self._lang},
                    )
                    _log_event({"type": "stt_result", "text": text})
                    break
                if text != last_partial:
                    last_partial = text
                    await bus.publish(
                        self._client, bus.CH_STT_PARTIAL,
                        {"text": text, "stable": False},
                    )
        finally:
            vad_task.cancel()
            try:
                await vad_task
            except asyncio.CancelledError:
                pass

    async def _vad_gate(
        self,
        frame_q: asyncio.Queue,
        out_q: asyncio.Queue,
    ) -> None:
        """Forward frames from `frame_q` to `out_q` until trailing silence
        exceeds `vad_min_silence_ms`. Then push None to signal end-of-utterance."""
        import numpy as np  # type: ignore
        import torch  # type: ignore

        model = self._vad["model"]
        # Silero expects float32 in [-1, 1] at 16 kHz.
        silence_frames_needed = max(
            1, self._vad_min_silence_ms // self._frame_ms
        )
        silence_run = 0
        # Pre-roll a little audio so the STT sees the start of the word.
        pad_frames = max(1, self._vad_speech_pad_ms // self._frame_ms)
        prebuf: deque[bytes] = deque(maxlen=pad_frames)
        speaking = False

        while True:
            pcm = await frame_q.get()
            arr = np.frombuffer(pcm, dtype="int16").astype("float32") / 32768.0
            with torch.no_grad():
                prob = float(model(torch.from_numpy(arr), self._sr).item())
            is_speech = prob >= 0.5

            if not speaking:
                prebuf.append(pcm)
                if is_speech:
                    speaking = True
                    for f in prebuf:
                        await out_q.put(f)
                    prebuf.clear()
                continue

            await out_q.put(pcm)
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= silence_frames_needed:
                    await out_q.put(None)
                    return


# ---------- Parakeet engine ----------


class ParakeetEngine:
    """NeMo Parakeet-TDT streaming wrapper.

    This is the seam where B2 (TensorRT INT8 export) lands. Today it uses
    the cache-aware streaming API straight from NeMo.
    """

    name = "parakeet"

    def __init__(self, model_name: str) -> None:
        from nemo.collections.asr.models import EncDecRNNTBPEModel  # type: ignore

        log.info("[voice/stt] loading %s ...", model_name)
        self._model = EncDecRNNTBPEModel.from_pretrained(model_name)
        self._model.eval()
        # Cache-aware streaming setup; concrete attribute names depend on the
        # NeMo version. We catch + log at first use so a bad version surfaces.
        self._stream_cfg_ready = False

    async def transcribe(
        self, frames: AsyncIterator[bytes]
    ) -> AsyncIterator[tuple[str, bool]]:
        # Implementation note: NeMo's streaming API is sync. We accumulate
        # frames into a buffer, run the model on chunked windows in a thread,
        # and yield partials. The exact streaming entrypoint differs across
        # NeMo versions; this is a placeholder that batches the full
        # utterance and emits a single final. Replace with the cache-aware
        # streaming call once B1's measurements are in.
        import numpy as np  # type: ignore

        chunks: list[bytes] = []
        async for f in frames:
            chunks.append(f)
        if not chunks:
            return
        pcm = b"".join(chunks)
        arr = np.frombuffer(pcm, dtype="int16").astype("float32") / 32768.0

        def _run() -> str:
            hyps = self._model.transcribe([arr])
            if not hyps:
                return ""
            h = hyps[0]
            return h.text if hasattr(h, "text") else str(h)

        text = await asyncio.to_thread(_run)
        yield text, True


# ---------- service ----------


def _select_backend(cfg: dict, client) -> object:
    mode = ((cfg.get("voice") or {}).get("mode") or "auto").lower()

    def try_mic():
        try:
            return MicBackend(client, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("mic backend unavailable (%s)", exc)
            return None

    if mode == "mic":
        b = try_mic()
        if b is None:
            raise SystemExit("voice.mode=mic but mic backend failed to load")
        return b
    if mode == "stdin":
        return StdinBackend(client)
    if mode == "null":
        return NullBackend(client)

    # auto
    b = try_mic()
    if b is not None:
        return b
    log.info("[voice] falling back to stdin backend")
    return StdinBackend(client)


async def run() -> int:
    cfg = _load_config()
    client = bus.get_client()
    backend = _select_backend(cfg, client)
    log.info("[voice] backend: %s", backend.name)
    try:
        await backend.run()
    finally:
        await client.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis voice capture")
    parser.add_argument(
        "--mode",
        choices=("auto", "mic", "stdin", "null"),
        help="override voice.mode from config",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose or os.environ.get("JARVIS_DEBUG") else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mode:
        os.environ["_JARVIS_VOICE_MODE_OVERRIDE"] = args.mode

        # Patch the loader to honour the override.
        global _load_config
        _orig = _load_config

        def _patched() -> dict:
            cfg = _orig()
            cfg.setdefault("voice", {})["mode"] = args.mode
            return cfg

        _load_config = _patched  # type: ignore[assignment]

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
