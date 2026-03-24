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

> **Note:** macOS is not supported — CUDA/TensorRT require an NVIDIA GPU.
