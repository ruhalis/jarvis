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

# torch2trt (TensorRT conversion) — tensorrt comes pre-installed with JetPack
cd vendor/torch2trt && python setup.py install && cd ../..

# whisper_trt
cd vendor/whisper_trt && python setup.py install && cd ../..

# Python dependencies
pip install -r requirements.txt
```

### Install on desktop (RTX GPU)

```bash
# PyTorch + TensorRT
pip install torch torchvision
pip install tensorrt

# torch2trt
cd vendor/torch2trt && python setup.py install && cd ../..

# whisper_trt
cd vendor/whisper_trt && python setup.py install && cd ../..

# Python dependencies
pip install -r requirements.txt
```

### CosyVoice TTS (optional — voice cloning)

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

# Test it
python cosy_voice.py test
```

> **Note:** macOS is not supported — CUDA/TensorRT require an NVIDIA GPU.
