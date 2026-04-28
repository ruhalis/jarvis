# RGB Matrix Display for Jarvis

Driving a Waveshare **RGB-Matrix-P3 64×64** (HUB75) panel from the Jetson Orin Nano Super to show idle clock/weather and listen/think/speak animations.

## TL;DR

Don't drive HUB75 directly from the Jetson. Offload to a small MCU (RP2040 or ESP32-S3) connected over USB serial. The Jetson publishes high-level state, the MCU renders.

## Why not drive it from the Jetson directly

The Waveshare P3 64×64 is a standard **HUB75 panel** — 16-pin ribbon, 5 V / ~3–4 A separate PSU, no on-board frame buffer. The host has to continuously bit-bang RGB + row-select + clock + latch + OE, with PWM in software for color depth. This needs hard real-time timing.

- `hzeller/rpi-rgb-led-matrix` is the gold-standard library, but it pokes Broadcom BCM2xxx registers directly and **does not work on Jetson**.
- `jetson-gpio` is a Python wrapper over sysfs/libgpiod — too slow and jittery for HUB75 at 64×64. Linux on the Orin Nano is not real-time, so you'd get visible flicker, tearing, and the panel would compete with whisper/TTS for CPU.
- No maintained HUB75 driver exists for Jetson. Adafruit's PioMatter / Pi 5 PIO trick relies on RP1 silicon the Orin doesn't have.

The Orin Nano's 8 GB RAM is already tight with WhisperTRT + Kokoro/Silero loaded; spending CPU on bit-banging a display is the wrong trade.

## Recommended architecture: display co-processor

```
Jetson Orin Nano                            MCU                    Panel
─────────────────                           ───                    ─────
Redis ──► jarvis_display.py ──USB serial──► Interstate 75 ──HUB75──► P3 64×64
          (state forwarder)                 or ESP32-S3
```

Two solid MCU options:

1. **Pimoroni Interstate 75 / 75 W** (RP2040 / RP2350) — purpose-built HUB75 driver, plugs into the panel's IDC connector directly, USB to the Jetson. ~$22. The 75 W has Wi-Fi if you want wireless. Uses RP2040 PIO + DMA, 10-bit gamma-corrected color.
2. **ESP32-S3 + HUB75 adapter** + the `mrcodetastic/ESP32-HUB75-MatrixPanel-DMA` library — DMA-driven, rock-solid 60+ Hz, Adafruit-GFX compatible, ~$5 for the adapter, easy to chain panels later.

Either solves the timing problem and frees the Jetson's CPU.

## Integration with Jarvis

Add a thin `jarvis_display.py` service that subscribes to existing Redis channels and forwards state to the MCU:

```
state_change (IDLE/LISTENING/PROCESSING/SPEAKING)  ─┐
tts_request / tts_done                              ├─►  jarvis_display.py  ──USB serial──►  MCU
weather/timer events                                ─┘
```

Protocol: keep it dumb — small JSON or tagged binary frames:

```json
{"mode": "listen"}
{"mode": "clock", "t": "14:32"}
{"mode": "weather", "temp": 12, "icon": "rain"}
{"mode": "speak", "level": 0.42}
```

All animation code lives on the MCU as sprite/shader routines. The Jetson just publishes state.

### State → animation mapping

| State | Animation |
|---|---|
| **IDLE** | Large clock, small weather icon + temp underneath, dim brightness |
| **LISTENING** | Pulsing ring or live mic-level bars (stream RMS from the VAD step) |
| **PROCESSING** | Rotating arc / Cylon scanner / boot-style spinner |
| **SPEAKING** | Animated mouth or EQ bars driven by TTS audio envelope (publish levels from `jarvis_tts.py`) |

## Bill of materials

- Waveshare RGB-Matrix-P3 64×64 (already have)
- **5 V / 4 A dedicated PSU** — do not power from the Jetson rail
- **Interstate 75 W** *or* **ESP32-S3 + HUB75 adapter** (~$5)
- 16-pin IDC ribbon (comes with the panel)
- Optional: 1000 µF cap across the panel's 5 V input to tame inrush

## File structure additions

```
~/.jarvis/services/
├── jarvis_display.py      # Redis subscriber → USB serial forwarder
└── firmware/
    └── jarvis_matrix/      # MCU firmware (Interstate 75 or ESP32)
        ├── main.py / .ino
        ├── animations/
        └── fonts/
```

Add to `config.yaml`:

```yaml
display:
  enabled: true
  port: /dev/ttyACM0       # or ttyUSB0 for ESP32
  baud: 115200
  brightness_idle: 40      # 0-255
  brightness_active: 180
```

## Build order

1. Wire panel + PSU + MCU on the bench, flash firmware with three hard-coded modes (`idle_clock`, `listen_pulse`, `think_spinner`). Verify no flicker at full brightness.
2. Add USB serial command parser on MCU (one JSON line = one state change).
3. Write `jarvis_display.py`, subscribe to `state_change`, map states to commands.
4. Add `speaking` waveform — publish TTS RMS levels from `jarvis_tts.py` at ~30 Hz.
5. Add weather + timer overlays once HA integration (Phase 3) lands.
6. Tie brightness to time of day (dim at night).

## Sources

- [hzeller/rpi-rgb-led-matrix](https://github.com/hzeller/rpi-rgb-led-matrix)
- [ESP32-HUB75-MatrixPanel-DMA](https://github.com/mrcodetastic/ESP32-HUB75-MatrixPanel-DMA)
- [Pimoroni Interstate 75](https://shop.pimoroni.com/products/interstate-75)
- [NVIDIA jetson-gpio](https://github.com/NVIDIA/jetson-gpio)
- [Adafruit PioMatter on Pi 5 (CNX Software)](https://www.cnx-software.com/2025/02/11/adafruit-piomatter-library-hub75-rgb-led-matrix-raspberry-pi-5/)
- [Waveshare RGB-Matrix-P3-64x64 wiki](https://www.waveshare.com/wiki/RGB-Matrix-P3-64x64)