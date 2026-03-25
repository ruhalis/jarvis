"""
=============================================================================
CosyVoice TTS Setup for Jarvis Assistant — Jetson Orin Nano Super (8GB)
=============================================================================

WHICH MODEL TO USE:
-------------------
For Jetson Orin Nano 8GB, use CosyVoice2-0.5B (best streaming support)
or CosyVoice-300M (lighter, if RAM is very tight).

Fun-CosyVoice3-0.5B is the highest quality but slightly heavier.
Start with CosyVoice2-0.5B — it has the best streaming architecture.

SETUP STEPS (run on Jetson):
----------------------------

# 1. Create conda environment
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# 2. Install PyTorch for Jetson (aarch64)
#    Get the correct wheel from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/
#    Example for JetPack 6.x:
pip install torch torchvision torchaudio  # Use Jetson-specific wheel!

# 3. Clone CosyVoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 4. Install dependencies
pip install -r requirements.txt
sudo apt-get install sox libsox-dev

# 5. Download model (pick ONE based on your RAM budget)
python -c "
from huggingface_hub import snapshot_download

# OPTION A: CosyVoice2-0.5B (RECOMMENDED — best streaming, ~3-4GB RAM)
snapshot_download('FunAudioLLM/CosyVoice2-0.5B',
                  local_dir='pretrained_models/CosyVoice2-0.5B')

# OPTION B: CosyVoice-300M (lighter, ~2-3GB RAM, older but proven)
# snapshot_download('FunAudioLLM/CosyVoice-300M',
#                   local_dir='pretrained_models/CosyVoice-300M')

# Text normalization resource (optional, skip ttsfrd .whl on ARM64)
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd',
                  local_dir='pretrained_models/CosyVoice-ttsfrd')
"

# NOTE: Do NOT install ttsfrd .whl on Jetson — it's x86_64 only.
#       CosyVoice will automatically fall back to 'wetext' which works fine.

# 6. Prepare your Jarvis voice reference WAV
#    - Extract 5-15 seconds of clean J.A.R.V.I.S. dialogue
#    - Must be: mono, 16kHz+, no background music/noise
#    - Save as: jarvis_en.wav (English) and jarvis_ru.wav (Russian dub)
"""

import sys
import os
import io
import time
import numpy as np
import torch
import torchaudio
import asyncio
import redis.asyncio as aioredis

# Add CosyVoice to path (vendored as git submodule)
COSYVOICE_ROOT = os.path.join(os.path.dirname(__file__), "vendor", "CosyVoice")
sys.path.append(COSYVOICE_ROOT)
sys.path.append(os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS"))


# =============================================================================
# 1. BASIC USAGE — Test that CosyVoice works
# =============================================================================

def test_basic():
    """Minimal test: generate speech and save to file."""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    # Load model (takes ~30s first time, uses GPU)
    model = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False,      # JIT not needed on Jetson
        load_trt=False,       # Enable later for TensorRT acceleration
        load_vllm=False,      # vLLM doesn't work on Jetson
        fp16=True,            # IMPORTANT: saves ~50% memory on Jetson
    )

    # --- English with built-in voice (no cloning) ---
    for i, chunk in enumerate(model.inference_zero_shot(
        tts_text="Good evening sir, all systems are operational.",
        prompt_text="Hope you can do better than me in the future.",
        prompt_speech_16k="./asset/zero_shot_prompt.wav",  # built-in sample
        stream=True,   # streaming = get audio chunks as they're ready
        speed=1.0,
    )):
        torchaudio.save(f"test_en_{i}.wav", chunk["tts_speech"], model.sample_rate)
        print(f"  Chunk {i}: {chunk['tts_speech'].shape[1] / model.sample_rate:.2f}s")

    print("English test done!")


# =============================================================================
# 2. VOICE CLONING — Clone Jarvis voice from your WAV
# =============================================================================

def test_jarvis_clone():
    """Clone Jarvis voice from reference WAV file."""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    model = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False, load_trt=False, load_vllm=False, fp16=True,
    )

    # Load YOUR Jarvis reference audio (5-15 seconds, clean, mono)
    jarvis_ref = load_wav("jarvis.wav", 16000)

    # --- Zero-shot clone: English ---
    # The prompt_text should be what's spoken in the reference audio
    # (helps the model understand the voice characteristics)
    for i, chunk in enumerate(model.inference_zero_shot(
        tts_text="Welcome home sir. Shall I run the usual evening routine?",
        prompt_text="The transcript of what jarvis says in your WAV file.",
        prompt_speech_16k=jarvis_ref,
        stream=True,
        speed=1.0,
    )):
        torchaudio.save(f"jarvis_en_{i}.wav", chunk["tts_speech"], model.sample_rate)

    # --- Cross-lingual clone: English voice speaking Russian ---
    # Uses the English Jarvis voice to speak Russian!
    for i, chunk in enumerate(model.inference_cross_lingual(
        tts_text="Добрый вечер, сэр. Все системы работают нормально.",
        prompt_speech_16k=jarvis_ref,
        stream=True,
        speed=1.0,
    )):
        torchaudio.save(f"jarvis_ru_{i}.wav", chunk["tts_speech"], model.sample_rate)

    print("Jarvis clone test done!")


# =============================================================================
# 3. PRE-CACHE SPEAKER EMBEDDING — Key optimization for latency
# =============================================================================

def cache_speaker_embedding():
    """
    Pre-compute and save speaker embedding at setup time.
    This avoids re-processing the reference WAV on every TTS call.
    Saves ~500ms per inference!
    """
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    model = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False, load_trt=False, load_vllm=False, fp16=True,
    )

    # Register and cache the Jarvis voice
    jarvis_ref = load_wav("jarvis.wav", 16000)

    # add_zero_shot_spk pre-computes the speaker embedding
    success = model.add_zero_shot_spk(
        prompt_text="The transcript of what jarvis says in your WAV file.",
        prompt_speech_16k=jarvis_ref,
        spk_id="jarvis_en"
    )
    assert success, "Failed to register speaker!"

    # Optionally register Russian voice too
    # jarvis_ru_ref = load_wav("jarvis_ru.wav", 16000)
    # model.add_zero_shot_spk("Транскрипт...", jarvis_ru_ref, "jarvis_ru")

    # Save all speaker embeddings to disk
    model.save_spkinfo()
    print("Speaker embeddings cached! Will auto-load on next startup.")


# =============================================================================
# 4. JARVIS TTS SERVICE — Production-ready with Redis Pub/Sub
# =============================================================================

class JarvisTTS:
    """
    TTS service for the Jarvis assistant.
    Integrates with your Redis Pub/Sub event bus.

    Subscribes to: llm_response (text to speak)
    Publishes to:  tts_audio (audio chunks), tts_done (finished speaking)
    """

    def __init__(
        self,
        model_dir: str = "pretrained_models/CosyVoice2-0.5B",
        default_voice: str = "jarvis_en",
        redis_url: str = "redis://localhost:6379",
    ):
        self.model_dir = model_dir
        self.default_voice = default_voice
        self.redis_url = redis_url
        self.model = None
        self.sample_rate = None

    async def initialize(self):
        """Load model and connect to Redis. Call once at startup."""
        print("[TTS] Loading CosyVoice2...")
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model = CosyVoice2(
            self.model_dir,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=True,
        )
        self.sample_rate = self.model.sample_rate
        print(f"[TTS] Model loaded. Sample rate: {self.sample_rate}")

        # Connect to Redis
        self.redis = aioredis.from_url(self.redis_url)
        print("[TTS] Connected to Redis")

    def detect_language(self, text: str) -> str:
        """Simple language detection based on character ranges."""
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        if cyrillic_count > len(text) * 0.3:
            return "ru"
        return "en"

    def synthesize_streaming(self, text: str, language: str = "auto") -> list:
        """
        Generate speech from text, yielding audio chunks.
        Returns list of numpy arrays (PCM float32).
        """
        if language == "auto":
            language = self.detect_language(text)

        chunks = []
        start = time.time()

        # Use cached speaker embedding (fast path)
        # If you cached with add_zero_shot_spk + save_spkinfo, do:
        for i, chunk in enumerate(self.model.inference_zero_shot(
            tts_text=text,
            prompt_text="",           # empty when using cached spk_id
            prompt_speech_16k="",     # empty when using cached spk_id
            zero_shot_spk_id=self.default_voice,
            stream=True,
            speed=1.0,
        )):
            audio_np = chunk["tts_speech"].squeeze().numpy()
            chunks.append(audio_np)

            if i == 0:
                first_chunk_time = time.time() - start
                print(f"[TTS] First chunk in {first_chunk_time*1000:.0f}ms")

        total_time = time.time() - start
        total_audio = sum(len(c) for c in chunks) / self.sample_rate
        print(f"[TTS] Done: {total_audio:.2f}s audio in {total_time*1000:.0f}ms "
              f"(RTF={total_time/total_audio:.3f})")

        return chunks

    def audio_to_bytes(self, audio_np: np.ndarray) -> bytes:
        """Convert float32 numpy to int16 PCM bytes for speaker output."""
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def run(self):
        """Main event loop: subscribe to llm_response, speak, publish tts_done."""
        await self.initialize()
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("llm_response")

        print("[TTS] Listening for llm_response events...")

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            text = message["data"].decode("utf-8")
            print(f"[TTS] Speaking: {text[:80]}...")

            # Notify state machine: now speaking
            await self.redis.publish("state_change", "SPEAKING")

            # Generate and stream audio (run sync TTS in thread to avoid blocking)
            chunks = await asyncio.to_thread(self.synthesize_streaming, text)

            for chunk_np in chunks:
                pcm_bytes = self.audio_to_bytes(chunk_np)
                # Publish raw PCM for the audio player service
                await self.redis.publish("tts_audio", pcm_bytes)

            # Done speaking
            await self.redis.publish("tts_done", "ok")
            await self.redis.publish("state_change", "IDLE")


# =============================================================================
# 5. FALLBACK CHAIN — Kokoro (fast) + CosyVoice (clone quality)
# =============================================================================

class HybridTTS:
    """
    Two-engine TTS: fast path (Kokoro) + quality path (CosyVoice).

    Use Kokoro for short confirmations: "Yes sir", "Right away", "Processing"
    Use CosyVoice for longer responses that benefit from Jarvis voice.

    This saves GPU memory — Kokoro is ~200MB ONNX vs CosyVoice ~3GB.
    Load CosyVoice on-demand, or keep both loaded if RAM allows.
    """

    def __init__(self):
        self.kokoro = None       # Lightweight, always loaded
        self.cosyvoice = None    # Loaded on demand or kept resident
        self.threshold = 50      # chars: below = Kokoro, above = CosyVoice

    async def speak(self, text: str, force_clone: bool = False):
        """Route to appropriate engine based on text length and context."""
        if len(text) < self.threshold and not force_clone:
            return self._speak_kokoro(text)
        else:
            return self._speak_cosyvoice(text)

    def _speak_kokoro(self, text: str):
        """Fast path: Kokoro ONNX, ~50-150ms, no voice cloning."""
        # Implementation depends on kokoro-onnx setup
        # See: https://github.com/thewh1teagle/kokoro-onnx
        pass

    def _speak_cosyvoice(self, text: str):
        """Quality path: CosyVoice streaming, Jarvis voice clone."""
        # Use JarvisTTS.synthesize_streaming() from above
        pass


# =============================================================================
# 6. PRE-GENERATED PHRASES — Zero latency for common responses
# =============================================================================

COMMON_PHRASES = {
    "en": [
        "Yes, sir.",
        "Right away, sir.",
        "Processing your request.",
        "All systems are operational.",
        "Good morning, sir.",
        "Good evening, sir.",
        "Welcome home, sir.",
        "As you wish, sir.",
        "I'll take care of it.",
        "Task completed.",
    ],
    "ru": [
        "Да, сэр.",
        "Сейчас, сэр.",
        "Обрабатываю запрос.",
        "Все системы работают.",
        "Доброе утро, сэр.",
        "Добрый вечер, сэр.",
        "Добро пожаловать домой, сэр.",
        "Как пожелаете, сэр.",
        "Я позабочусь об этом.",
        "Задача выполнена.",
    ]
}


def pregenerate_phrases():
    """
    Run ONCE at setup: generate WAV files for all common phrases.
    These play instantly (0ms latency) — no TTS needed at runtime.
    """
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    os.makedirs("cache/phrases/en", exist_ok=True)
    os.makedirs("cache/phrases/ru", exist_ok=True)

    model = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False, load_trt=False, load_vllm=False, fp16=True,
    )

    jarvis_ref = load_wav("jarvis.wav", 16000)

    for lang, phrases in COMMON_PHRASES.items():
        for phrase in phrases:
            safe_name = phrase[:40].replace(" ", "_").replace(",", "").replace(".", "")
            outpath = f"cache/phrases/{lang}/{safe_name}.wav"

            if os.path.exists(outpath):
                continue

            print(f"[Cache] Generating: {phrase}")
            all_audio = []

            if lang == "en":
                gen = model.inference_zero_shot(
                    tts_text=phrase,
                    prompt_text="Transcript of jarvis reference.",
                    prompt_speech_16k=jarvis_ref,
                    stream=False,
                )
            else:
                gen = model.inference_cross_lingual(
                    tts_text=phrase,
                    prompt_speech_16k=jarvis_ref,
                    stream=False,
                )

            for _, chunk in enumerate(gen):
                all_audio.append(chunk["tts_speech"])

            full_audio = torch.cat(all_audio, dim=1)
            torchaudio.save(outpath, full_audio, model.sample_rate)
            print(f"  Saved: {outpath}")

    print("\nAll common phrases pre-generated!")


# =============================================================================
# ENTRYPOINTS
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Jarvis TTS Service")
    parser.add_argument("command", choices=[
        "test",           # Test basic TTS
        "clone",          # Test Jarvis voice clone
        "cache-voice",    # Pre-compute speaker embeddings
        "cache-phrases",  # Pre-generate common phrases
        "serve",          # Run TTS service (Redis pub/sub)
    ])
    args = parser.parse_args()

    if args.command == "test":
        test_basic()
    elif args.command == "clone":
        test_jarvis_clone()
    elif args.command == "cache-voice":
        cache_speaker_embedding()
    elif args.command == "cache-phrases":
        pregenerate_phrases()
    elif args.command == "serve":
        tts = JarvisTTS()
        asyncio.run(tts.run())