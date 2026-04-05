# Local TTS engines for a Jarvis assistant on Jetson Orin Nano

**XTTS v2 is the only open-source engine that combines Russian support, English support, and zero-shot voice cloning — but it consumes ~5 GB of the Jetson's 8 GB shared memory.** The practical solution is a hybrid architecture: Kokoro-82M (ONNX, ~100 MB) for fast British English responses, Silero v5 (~85 MB) for high-clarity Russian, and XTTS v2 with pre-cached speaker embeddings reserved for voice-cloned output when quality matters most. This three-engine stack fits within the Orin Nano's memory budget, can hit **sub-200 ms latency** on the fast path, and covers both languages with cloning capability. No single engine solves every requirement, but the right combination gets remarkably close.

---

## The Russian language requirement eliminates most contenders

Of the 18+ engines evaluated, **only three support Russian TTS at all**: XTTS v2 (17 languages, voice cloning), Silero v5 (Russian-first, no cloning), and Piper (4 Russian VITS voices, no cloning). Every other engine — Kokoro, StyleTTS 2, Fish Speech, GPT-SoVITS, F5-TTS, Zonos, Mars5, Orpheus, Parler-TTS, Dia, Spark TTS, MeloTTS, Matcha-TTS, MetaVoice, WhisperSpeech, OuteTTS — lacks Russian entirely or has only experimental coverage. This single constraint collapses the decision space dramatically.

For Russian voice *cloning* specifically, XTTS v2 is effectively the only viable option. Fish Speech lists Russian as "Tier 2" but requires **4+ GB VRAM** and has weaker Russian pronunciation. NVIDIA Riva's Magpie TTS supports zero-shot cloning but **does not include Russian** in its language set (en, es, fr, de, zh, vi, it only). Alpha Cephei's independent Russian TTS benchmark found XTTS v2's Russian CER is **2.7** — significantly worse than Silero's **0.7** — meaning cloned Russian speech will be intelligible but noticeably less crisp than a dedicated Russian engine.

---

## Engine-by-engine comparison for Jetson deployment

### Tier 1: Best fit for Jetson Orin Nano 8 GB

| Engine | Params | RAM | Latency (Jetson est.) | English | Russian | Voice clone | Streaming | License |
|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | 82M | ~200–400 MB | **50–150 ms** (GPU) | ✅ 8 British voices | ❌ | ❌ | ✅ | Apache 2.0 |
| **Piper TTS** | 15–100 MB models | **50–150 MB** | **<100 ms** (CPU!) | ✅ en_GB voices | ✅ 4 voices | ⚠️ Via training | ✅ raw PCM | MIT |
| **Silero v5** | ~85 MB | **100–200 MB** | **100–300 ms** (CPU) | ✅ adequate | ✅ **best clarity** | ❌ | ⚠️ sentence-level | CC-BY-NC-SA |

**Kokoro-82M** is the standout for English. Despite being only 82M parameters, it ranked **#1 on HuggingFace's TTS Arena**, outperforming models 5–15× its size. It offers **8 British English voices** — `bm_george` and `bm_fable` are the best male British options for a butler aesthetic, and voice blending (`bm_george:0.7,bm_fable:0.3`) enables custom timbres. The ONNX model comes in fp32 (~300 MB), fp16, q8, and **q4 (~80 MB)** variants. Pre-built ARM64 Docker images exist in the jetson-containers ecosystem (`kokoro-tts:onnx`). On desktop GPUs it achieves **210× real-time**; on Jetson Orin Nano's 1024 CUDA cores, estimated **50–150 ms** for short sentences easily hits the 200 ms target.

**Piper TTS** is the ultimate lightweight fallback — designed for Raspberry Pi, it runs comfortably on the Orin Nano's ARM CPU alone at RTF ~0.2. Pre-built ARM64 binaries ship ready to go. For Russian, `ru/irina` achieved the **highest UTMOS score (3.67)** in the Alpha Cephei benchmark despite using only 1 hour of training data. English `en_GB/cori` (high quality) works for a British voice. The entire model + runtime uses **under 150 MB RAM**. A community-proven approach on Jetson: persistent Piper process with raw PCM piped to PulseAudio, with a startup warmup sentence to eliminate cold-start latency.

**Silero v5** is purpose-built for Russian. Automated stress placement and homograph resolution — critical for Russian prosody — set it apart. Six Russian speakers, 8/24/48 kHz output, SSML support. At RTF **0.054 on CPU** (v4), it's fast enough without GPU acceleration. The v5 release (October 2025) added quality improvements. The CC-BY-NC-SA license restricts commercial use, but for a personal Jarvis assistant this is irrelevant.

### Tier 2: Voice cloning capable, tight fit on Jetson

| Engine | Params | RAM | Latency | Russian | Clone quality | Streaming | Jetson viable? |
|---|---|---|---|---|---|---|---|
| **XTTS v2** | ~467M | **~5 GB** (unified) | 500 ms–2 s | ✅ (CER 2.7) | ✅ zero-shot 6 s | ✅ <200 ms chunks | ⚠️ tight |
| **GPT-SoVITS v4** | 167–407M | ~4 GB | RTF **0.028** (desktop) | ❌ | ✅ zero-shot 5 s | ✅ | ✅ size, ❌ no Russian |
| **Fish Speech 1.5** | ~500M | ~4+ GB | <150 ms streaming | ⚠️ Tier 2 | ✅ zero-shot 10 s | ✅ | ⚠️ marginal |

**XTTS v2** remains the most versatile voice cloning engine for this use case. Zero-shot cloning from **6 seconds** of reference audio across 17 languages, with built-in streaming delivering **<200 ms to first audio chunk** on consumer GPUs. The critical optimization is **pre-computing speaker embeddings** at setup time and caching them to disk — this eliminates the ~500 ms–1 s reference audio processing step from every inference call. On the Jetson's 8 GB shared memory, XTTS v2 consumes approximately **5 GB** during inference (1.87 GB model weights + memory for GPT block initialization), leaving ~3 GB for the OS and other lightweight processes. This means XTTS v2 cannot coexist with a local LLM — the LLM must be offloaded to a cloud API when voice cloning is active. Coqui AI shut down in early 2024, but the Idiap Research Institute maintains the fork at `idiap/coqui-ai-TTS`. The **non-commercial Coqui Public Model License** applies.

**GPT-SoVITS v4** deserves mention for its extraordinary efficiency: only **167M parameters** (v2) with RTF **0.014 on RTX 4090**, official ARM64 Docker support, and MIT license. If Russian support were added (there's an open GitHub issue), this would be the ideal Jetson engine. It's worth monitoring.

### Tier 3: Promising but impractical for this hardware/use case

| Engine | Why not viable |
|---|---|
| **Orpheus TTS** (3B) | Excellent quality + emotion, but no Russian. 3B Q4 GGUF ≈ 2–3 GB via llama.cpp — possible on Jetson but latency likely 500 ms+ for 3B. Smaller 1B/400M models announced but not yet released. |
| **Spark TTS** (0.5B) | Outstanding hardware fit (tiny Qwen2.5 model, TensorRT-LLM path), but Chinese + English only, no Russian. |
| **F5-TTS** (335M) | Good voice cloning, "state-of-the-art" quality, but no Russian, no streaming, and **6–8 GB VRAM** on desktop translates to near-total memory consumption on Jetson. |
| **StyleTTS 2** | Highest MOS scores for English (4.1+ on LJSpeech), but English only, no streaming, no Russian. |
| **Zonos** (1.6B) | Rich emotion control but 1.6B params requires 6+ GB VRAM. No Russian. |
| **Dia TTS** (1.6B) | Impressive multi-speaker dialogue but ~10 GB VRAM, English only. Dia2 1B may fit when quantized. |
| **Mars5-TTS** (1.2B) | English only in open-source version. AGPL license. Stale repo. |
| **MetaVoice** (1.2B) | Requires **≥24 GB GPU RAM** per documentation. Completely infeasible. |
| **Parler-TTS** | Text-described voice control is innovative but no voice cloning from audio, English only. |
| **MeloTTS** | Has British accent option, but no Russian, no voice cloning, ~600 MB due to BERT dependency. |
| **Matcha-TTS** | Exceptional RTF (0.015), but English-only single speaker, no cloning. |
| **WhisperSpeech** | Stalled development, no Russian, uncertain memory requirements. |
| **OuteTTS** | GGUF support is nice, but **latency is catastrophic** — over 4 minutes for 200 words on the 1B model. |

---

## NVIDIA Riva and Jetson-native optimization paths

**NVIDIA Riva** offers the most polished Jetson TTS experience with its Magpie models, including zero-shot voice cloning (Magpie Zeroshot and Magpie Flow, using ~5-second reference clips). TensorRT-optimized models deliver claimed **<100 ms latency** on Jetson Orin. However, three significant blockers exist: **no Russian TTS** (only en, es, fr, de, zh, vi, it), the full Riva stack consumes **~6.6 GB RAM** (ASR+TTS together per JetsonHacks measurements), and it requires an **NVIDIA AI Enterprise license** for production use (free 90-day trial available). If you enable TTS-only and disable ASR/NLP/NMT services, RAM drops to an estimated 3–5 GB, but this still leaves no room for other workloads. Multiple forum users report model-loading failures on Orin Nano due to memory pressure.

**TensorRT optimization** is the most impactful acceleration path for TTS on Jetson. The `jetson-voice` project by Dusty Franklin demonstrates FastPitch + HiFi-GAN with TensorRT achieving **126–232 ms time-to-first-audio** and **21–23× real-time** throughput after warmup. The conversion workflow is straightforward: export to ONNX, then run `trtexec --onnx=model.onnx --saveEngine=model.trt --fp16`. Engines must be built on the target Jetson device (compute capability 8.7). FP16 inference is the recommended default — Ampere Tensor Cores provide ~2× throughput over FP32 with negligible quality loss. INT8 requires a calibration dataset and shows limited benefit for CNN-heavy vocoders.

**ONNX Runtime with CUDA Execution Provider** is confirmed working on Jetson but requires special installation — standard `pip install onnxruntime-gpu` fails on aarch64. Three paths exist: pre-built Docker containers from `jetson-containers`, the Jetson AI Lab PyPI index (`pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.dev`), or building from source. Both Kokoro and Piper ship native ONNX models, making this the most practical acceleration path. Verify installation with `onnxruntime.get_available_providers()` — it should show `TensorrtExecutionProvider` and `CUDAExecutionProvider`.

The **jetson-containers** project provides pre-built Docker images for `kokoro-tts:onnx`, `piper-tts`, `piper1-tts:1.3.0`, and `onnxruntime` with GPU support — these are the fastest path to deployment.

---

## Voice cloning from a single WAV file: what actually works

For zero-shot cloning, **10–15 seconds of clean reference audio is the sweet spot**. Quality scales roughly linearly from 3→15 seconds, then plateaus. Longer clips (>30 seconds) can actually degrade generation quality — Qwen3-TTS documentation explicitly warns that excessively long references cause generation hangs. The reference WAV should be 16-bit PCM, mono, ≥24 kHz sample rate, with no background noise, music, or reverb, and valid speech occupying at least 60% of the duration.

For the **Jarvis voice specifically**, extract 10–15 seconds of clean J.A.R.V.I.S. dialogue from Iron Man films for zero-shot cloning via XTTS v2. Expected speaker similarity will be **~0.65–0.75** — capturing the general British, calm, articulate tone but not a perfect match. For substantially better results, compile **30+ minutes** of Paul Bettany's JARVIS dialogue and fine-tune XTTS v2, which should push similarity to ~0.85+. Alternatively, fine-tune a Piper model on JARVIS data for a dedicated fast voice — this eliminates the need for the heavy XTTS v2 at runtime. Legal risk for personal, non-commercial use is minimal: voice characteristics are not copyrightable (per *Lehrman v. Lovo*, 2025), though distributing the model or using it commercially would raise right-of-publicity concerns.

**The single most important optimization for voice cloning latency on Jetson** is pre-computing and caching speaker embeddings:

```python
# Run once at setup time:
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["jarvis_reference.wav"], gpt_cond_len=30, max_ref_length=60
)
# Serialize to disk as JSON/tensor file

# At runtime — no reference audio processing needed:
audio_stream = model.inference_stream(text, "en", gpt_cond_latent, speaker_embedding)
```

This eliminates the ~500 ms–1 s reference audio processing step from every inference call. Store multiple profiles (Jarvis English, Russian voice) and load the appropriate one at boot.

For **Russian voice cloning**, XTTS v2 with `language="ru"` is the only practical open-source option. Expect lower fidelity than English cloning (CER 2.7 vs English CER ~1.0). For high-quality Russian without cloning, Silero v5's dedicated Russian speakers will sound substantially more natural.

---

## Recommended architecture for Jetson Orin Nano 8 GB

The optimal design uses three TTS engines loaded selectively, with a router that picks the right engine based on language and response type:

| Function | Engine | RAM | Latency | When to use |
|---|---|---|---|---|
| **English fast path** | Kokoro-82M ONNX (q8) | ~200 MB | **50–150 ms** | Short confirmations, status updates, <50 chars |
| **English cloned voice** | XTTS v2 (FP16, cached embeddings) | ~5 GB | **200–500 ms** streaming first chunk | Longer responses, when Jarvis voice matters |
| **Russian fast path** | Silero v5 | ~100 MB | **100–300 ms** | All Russian responses without cloning |
| **Russian cloned voice** | XTTS v2 (`language="ru"`) | ~5 GB | **500 ms–1 s** | Russian responses needing cloned voice |

Kokoro + Silero can coexist in memory (~300 MB combined), leaving ample room for STT and a small local LLM. When XTTS v2 is needed for cloned voice output, either offload the LLM to a cloud API or unload Kokoro/Silero temporarily. Pre-generate common phrases ("Yes, sir," "Right away," "Processing your request") as cached WAV files for **instant zero-latency playback**.

The critical **pre-built British voice** recommendation: Kokoro's `bm_george` is the closest to a Jarvis-like butler voice available without cloning. Blending `bm_george:0.7,bm_fable:0.3` can further refine the tone. For Russian, Silero v5's `xenia` (female) or `aidar`/`eugene` (male) speakers deliver the best clarity among all open-source Russian options.

---

## Conclusion

The Jetson Orin Nano's 8 GB shared memory is the binding constraint, not compute. Three engines cover all requirements: **Kokoro-82M for fast, high-quality British English** (the best quality-per-byte in the TTS landscape), **Silero v5 for crisp Russian**, and **XTTS v2 for voice cloning in both languages** when loaded on demand. The fastest deployment path uses `jetson-containers` Docker images with ONNX Runtime CUDA EP for Kokoro and Piper, avoiding manual dependency management. Pre-computing XTTS v2 speaker embeddings and caching common phrases as WAV files are the two highest-impact latency optimizations. Monitor GPT-SoVITS for potential Russian support and Orpheus TTS's forthcoming 400M/150M models — either could collapse the multi-engine stack into a single solution. For the Jarvis voice, start with Kokoro's `bm_george` British voice for immediate results, then fine-tune a Piper model on extracted JARVIS dialogue for a dedicated fast clone that bypasses XTTS v2 entirely at runtime.