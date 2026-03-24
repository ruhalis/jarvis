"""
Real-time microphone transcription using WhisperTRT 
"""
import argparse
import sys
import time
from collections import deque
from threading import Event

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION_MS = 30  # ms per audio callback block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# VAD thresholds
ENERGY_THRESHOLD = 0.01  # RMS energy to consider as speech
SILENCE_DURATION = 0.8   # seconds of silence before transcribing
MIN_SPEECH_DURATION = 0.3  # minimum speech length to transcribe

# Keep up to 30 seconds of audio
MAX_BUFFER_SECONDS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="WhisperTRT mic transcriber")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small)")
    parser.add_argument("--device-index", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--energy-threshold", type=float, default=ENERGY_THRESHOLD,
                        help="RMS energy threshold for speech detection")
    return parser.parse_args()


def list_devices():
    print(sd.query_devices())


def load_model(model_name: str):
    from whisper_trt import load_trt_model
    print(f"Loading WhisperTRT model '{model_name}' (first load builds TensorRT engine, may take a few minutes)...")
    model = load_trt_model(model_name)
    print("Model loaded.")
    return model


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    model = load_model(args.model)

    audio_buffer = deque(maxlen=int(MAX_BUFFER_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    is_speaking = False
    silence_blocks = 0
    speech_blocks = 0
    silence_blocks_needed = int(args.energy_threshold and SILENCE_DURATION * 1000 / BLOCK_DURATION_MS)
    min_speech_blocks = int(MIN_SPEECH_DURATION * 1000 / BLOCK_DURATION_MS)
    stop_event = Event()

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, silence_blocks, speech_blocks

        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        chunk = indata[:, 0].copy()  # mono
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms >= args.energy_threshold:
            if not is_speaking:
                is_speaking = True
                silence_blocks = 0
                speech_blocks = 0
            speech_blocks += 1
            silence_blocks = 0
            audio_buffer.append(chunk)
        elif is_speaking:
            silence_blocks += 1
            audio_buffer.append(chunk)
            if silence_blocks >= silence_blocks_needed:
                if speech_blocks >= min_speech_blocks:
                    audio = np.concatenate(list(audio_buffer))
                    transcribe(model, audio)
                audio_buffer.clear()
                is_speaking = False
                silence_blocks = 0
                speech_blocks = 0

    def transcribe(model, audio: np.ndarray):
        audio = audio.astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        t0 = time.monotonic()
        result = model.transcribe(audio)
        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()
        if text:
            print(f"[{duration:.1f}s audio, {elapsed:.1f}s inference] {text}")

    print(f"Listening on device: {args.device_index or 'default'} | "
          f"Sample rate: {SAMPLE_RATE} Hz | Energy threshold: {args.energy_threshold}")
    print("Speak into the microphone. Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            device=args.device_index,
            callback=audio_callback,
        ):
            stop_event.wait()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
