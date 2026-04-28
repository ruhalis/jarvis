# jarvis

## Setup

```bash
git clone --recursive https://github.com/<your-user>/jarvis.git
# or
git clone --recursive git@github.com:<your-user>/jarvis.git
cd jarvis
python -m venv .venv
source .venv/bin/activate
```

### Install on Jetson Orin Nano (JetPack 6.2)

```bash
# PyTorch for JetPack 6.2
pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Python dependencies
pip install -r requirements.txt
```

STT uses Parakeet-TDT-0.6b-v3 via NeMo (TensorRT export through NeMo's
own tooling). See `TECHNICAL.md` for the full pipeline.

### CosyVoice TTS (optional — voice cloning, Phase 5)

```bash
# Install CosyVoice dependencies (from vendored submodule)
cd vendor/CosyVoice
pip install -r requirements.txt
cd ../..

# System dependency
sudo apt-get install sox libsox-dev

# Download model (~2GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B',
                  local_dir='vendor/CosyVoice/pretrained_models/CosyVoice2-0.5B')
"

# NOTE: Do NOT install the ttsfrd .whl on Jetson — it's x86_64 only.
# CosyVoice falls back to wetext automatically.
```

> **Note:** macOS is not supported — CUDA/TensorRT require an NVIDIA GPU.
