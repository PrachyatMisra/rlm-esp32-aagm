# RLM + AAGM on ESP32

Hardware track of the RLM research project: the Recursive Language Model with
**Asynchronous Adaptive Gating Mechanism (AAGM)** running on an ESP32 DevKit
(Arduino framework). The full internet analysis behind the design choices is in
[`docs/RESEARCH.md`](docs/RESEARCH.md). Hardware-free verification details are
in [`verification/README.md`](verification/README.md).

## What is in this directory

```
esp32/
  tools/                      # host-side (Python) - training, quantisation, export
    corpus.py                 # HF imdb loader + offline lexicon-bootstrapped corpus
    aagm.py                   # CompactRLM + ACT halting + shadow halt-forecaster
    quantize.py               # symmetric per-channel int8 (LiteRT-style, zp=0)
    train_export.py           # trains, calibrates tau_lo, quantises, exports
                              # firmware headers + golden vectors + artifacts
    out/golden_vectors.txt    # int8-mirror expectation vectors for the harness
  firmware/
    rlm_esp32/
      rlm_esp32.ino           # Arduino sketch: dual-core arbiter + serial protocol
      platformio.ini          # optional PlatformIO build (esp32dev / esp32-s3)
      src/
        rlm_engine.h/.cpp     # portable inference engine (no Arduino deps)
        tokenizer.cpp         # FNV-1a hash tokenizer (byte-identical to Python)
        rlm_config.h          # GENERATED: dims + halting constants
        rlm_weights.h         # GENERATED: int8 weights (510 KiB) + f32 params
  test/
    host_harness.cpp          # native parity: real engine vs golden vectors
    Makefile
  verification/verilog/       # optional Icarus Verilog control-policy check
  host/
    gateway.py                # serial client, on-device eval, --selftest parity
  docs/
    RESEARCH.md               # extensive internet analysis with citations
```

## The architecture at a glance

- **Model**: weight-shared recursive block (LN -> 4-head MHA -> residual -> LN ->
  GELU FFN -> residual -> 0.1 state refinement), hidden 96, sequence <= 96 tokens,
  max 8 recursion steps, classifier head. 524 K float params.
- **ACT halting (software AAGM)**: p_k = 1 - g_k; halt after step k >= 1 when
  sum(1 - g) >= 0.9. Trained with a small ponder-cost term so depth commits.
- **Shadow gate (software+hardware AAGM)**: 97-param forecaster one step ahead of a
  halt, calibrated to >= 97% precision; a pure scheduler hint (never changes logits).
- **Async arbiter (hardware AAGM)**: FreeRTOS task on core 0 evaluates all gates via
  a zero-copy mailbox while core 1 refines; also owns battery sampling + serial.
- **Energy budget gate (hardware AAGM)**: PERF 240 MHz/budget 8, BAL 160 MHz/6,
  ECO 80 MHz/4; battery-driven AUTO mode; mid-inference budget shrink aborts at
  cooperative checkpoints (`trace.aborted`).
- **Tokenizer**: `id = FNV1a(word) % 4095 + 1`, PAD=0; no vocab table on device.

Measured software-side (see artifacts for the full record):

| configuration | accuracy | F1 | mean steps | note |
|---|---|---|---|---|
| AAGM + ACT | 0.9233 | 0.9201 | 2.18 / 8 | adaptive halting |
| fixed depth 8 | 0.9233 | 0.9201 | 8.00 | ablation |
| int8 mirror (= firmware) | 0.9233 | 0.9201 | 2.18 | per-channel int8 |

Firmware parity (host-native build of the shipping C++ engine): 12/12 golden
vectors, max |logit err| 4.5e-6, decisions/steps/tokenization exact.

## Build & flash (Arduino IDE)

1. Install the **ESP32 Arduino core** (Boards Manager: "esp32 by Espressif", 2.x/3.x).
2. Open `firmware/rlm_esp32/rlm_esp32.ino` in the Arduino IDE.
3. Select your board (e.g. "DOIT ESP32 DEVKIT V1" or "ESP32S3 Dev Module") and port.
4. Upload. Serial monitor at **115200 baud**, line ending = "Newline".

Or with PlatformIO: `cd firmware/rlm_esp32 && pio run -t upload && pio device monitor`.

## Serial protocol

```
INFER <text>      -> {"pred":1,"logits":[..],"steps":2,"depth":..,"gates":[..],
                      "shadows":[..],"hints":1,"aborted":0,"us":..,"budget":8,...}
MODE PERF|BAL|ECO|AUTO      MODE AUTO derives profile from battery level
BATT <millivolts>|AUTO      BATT AUTO: GPIO34 100k/100k divider (x2); else simulated mV
BENCH <n>         -> timing stats over n runs of a fixed review
STAT              -> heap, cpu MHz, budget, board
PING              -> {"pong":1,...}
```

Host automation: `python host/gateway.py --port /dev/ttyUSB0 --eval`
(pyserial). Without hardware: `python host/gateway.py --selftest` compiles the
shipping engine natively and re-checks parity. The complete no-board workflow
is documented in [`verification/README.md`](verification/README.md).

## Retraining / regenerating weights

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch numpy pyserial matplotlib
cd esp32/tools
python train_export.py                 # offline-friendly; --dataset hf when online
cd ../firmware/test && make check      # native parity vs new golden vectors
```

This rewrites `firmware/rlm_esp32/src/rlm_{config,weights}.h` - reflash afterwards.

## Wiring notes

- No peripherals are required for the NLP demo itself.
- Battery telemetry is optional: `BATT AUTO` reads GPIO34 through a 100k/100k
  divider from your Li-ion pack (max 4.2 V -> 2.1 V at the pin). If the pin floats,
  firmware falls back to a simulated 4000 mV and says so in `STAT.batt_src`.
- Power profiling: for *measured* mA (vs the model-based estimates in
  `artifacts/esp32_energy_model.png`), put an INA219 (I2C) between USB 5 V and the
  board and log during `BENCH 50` in each MODE.
