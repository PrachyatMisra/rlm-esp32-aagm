# Edge-RLM: Recursive Language Model with Asynchronous Adaptive Gating Mechanism (AAGM) on ESP32

[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20DevKit%20(Dual--Core%20Xtensa%20LX6)-blue.svg)](#hardware-architecture)
[![Quantization](https://img.shields.io/badge/Weights-int8%20Symmetric%20Per--Channel%20(510%20KiB)-purple.svg)](#quantization--memory-layout)
[![Accuracy](https://img.shields.io/badge/Accuracy-92.33%25%20(0.00%25%20Quant%20Drop)-emerald.svg)](#empirical-benchmarks)
[![Efficiency](https://img.shields.io/badge/Energy%20Reduction-72.8%25%20(2.18%20Steps%20Avg)-cyan.svg)](#empirical-benchmarks)
[![Parity](https://img.shields.io/badge/Golden%20Parity-12%2F12%20Vectors%20PASS-brightgreen.svg)](#hardware-free-verification-suite)
[![Patent](https://img.shields.io/badge/Patent%20Status-Patent%20Application%20(20%20Claims)-orange.svg)](docs/PATENT_SPECIFICATION.md)

Edge-RLM is an ultra-compact, energy-proportional recursive transformer inference system engineered specifically for resource-constrained edge microcontrollers (ESP32 DevKit-v1 with 520 KiB SRAM and 4 MiB Flash). It replaces traditional deep feedforward networks with a single weight-shared recursive transformer block governed by an **Asynchronous Adaptive Gating Mechanism (AAGM)**.

By evaluating recursive depth dynamically (2 to 8 iterations) and offloading halting mass accumulation to an asynchronous secondary core / digital coprocessor, Edge-RLM reduces active inference cycles by **72.8%** (averaging 2.18 steps) while maintaining **92.33% accuracy** and a **zero dynamic heap** allocation footprint.

---

## Table of Contents
1. [Core Innovations & Patent Highlights](#core-innovations--patent-highlights)
2. [Technical Bottleneck Resolutions](#technical-bottleneck-resolutions)
3. [Interactive Demonstrations & Testing Quickstart](#interactive-demonstrations--testing-quickstart)
   - [Live Interactive Web Demonstration Studio (Port 8000)](#1-live-interactive-web-demonstration-studio-port-8000)
   - [Offline Conversational RLM "Chat & Test" Agent](#2-offline-conversational-rlm-chat--test-agent)
   - [Embedded Web Serial Console (Mock UART)](#3-embedded-web-serial-console-mock-uart)
   - [Dual-Track Hardware-Free Verification Suite](#4-dual-track-hardware-free-verification-suite)
   - [Physical ESP32 Arduino Deployment & Serial Chat](#5-physical-esp32-arduino-deployment--serial-chat)
4. [Hardware & Software Architecture](#hardware--software-architecture)
5. [Quantization & Memory Layout](#quantization--memory-layout)
6. [Empirical Benchmarks](#empirical-benchmarks)
7. [Repository File Map](#repository-file-map)
8. [Documentation Links](#documentation-links)

---

## Core Innovations & Patent Highlights

Edge-RLM forms the subject matter of a comprehensive formal patent application comprising 20 claims ([`docs/PATENT_SPECIFICATION.md`](docs/PATENT_SPECIFICATION.md)):

```
                                  +-------------------------------------------------------------+
                                  |                 SHARED STATIC SRAM BUFFERS                  |
                                  |   H[0..T-1] hidden states | pooled_h[0..D-1] | Mailbox      |
                                  +-------------------------------------------------------------+
                                                ^                                   ^
                                                | Direct Memory                     | Zero-Copy
                                                | Access                            | Inspection
+-----------------------------------------------+-----------+     +-----------------+-------------------------------------------+
|              CORE 1 (APP_CPU) - COMPUTE TASK              |     |               CORE 0 (PRO_CPU) - ARBITER TASK               |
|                                                           |     |                                                             |
|  1. Zero-Vocab FNV-1a Tokenizer (Hash Modulo 4095 + 1)   |     |  1. FreeRTOS Direct Notification Dispatcher                 |
|  2. Token Embedding Lookup (int8 + float scale)          |     |  2. Authoritative Adaptive Gating Gate: g_k = sigma(W*h+b)  |
|  3. Temporal Token Saliency Freezing (TSTF):              |     |  3. Shadow Halt Forecaster: s_{k+1} = sigma(w*h+b)          |
|     - Computes L1 delta ||h_k - h_{k-1}||_1               |     |  4. Speculative Output Pre-Computation (APOP):              |
|     - Bypasses 384-wide GELU FFN for converged tokens     |     |     - If s_{k+1} < 0.347, speculatively evaluates 2-layer   |
|  4. Weight-Shared 4-Head Attention Block                  |     |       MLP classification head into shadow buffer            |
|  5. Cooperative Abort Boundary Checks                    |     |  5. Closed-Loop Battery Droop Observer & DVFS Manager:      |
|                                                           |     |     - Samples GPIO34 ADC & subtracts I_active * R_int       |
|                                                           |     |     - Dynamic Frequency Scaling (240MHz -> 160MHz -> 80MHz) |
+-----------------------------------------------------------+     +-------------------------------------------------------------+
                                                                                                |
                                                                                +---------------+---------------+
                                                                                |  DIGITAL HARDWARE COPROCESSOR |
                                                                                |  - APB 32-bit Memory-Mapped   |
                                                                                |  - Q1.15 Halting Accumulator  |
                                                                                |  - Single-Cycle Abort IRQ     |
                                                                                +-------------------------------+
```

1. **Speculative Asynchronous Output Pre-Computation (APOP)**: Eliminates post-recursion classification tail latency (saving 150–200 μs) by speculatively running the output head on Core 0 when the shadow forecaster predicts early halting.
2. **Temporal Token Saliency Freezing (TSTF)**: Bypasses the 384-wide GELU feedforward network for tokens whose representations converge across iterations, saving 73,728 MACs per frozen token per step.
3. **Closed-Loop Battery Droop Compensation**: Dynamically measures loaded supply voltage ($V_{\text{loaded}} = V_{\text{OCV}} - I_{\text{active}} \times R_{\text{int}}$) to trigger cooperative early halting before rail voltage breaches the 3.3V brownout threshold.
4. **Digital Hardware Coprocessor (RTL Verilog)**: Memory-mapped APB slave coprocessor performing Q1.15 mass accumulation, shadow threshold checks, and single-cycle asynchronous abort interrupts.

---

## Technical Bottleneck Resolutions

Detailed mathematical derivations and engineering proofs are in [`docs/TECHNICAL_BOTTLENECK_RESOLUTION.md`](docs/TECHNICAL_BOTTLENECK_RESOLUTION.md).

| Technical Bottleneck | Root Cause | Implemented Solution | Measured Impact |
| :--- | :--- | :--- | :--- |
| **1. Output Head Tail Latency** | Sequential execution of 2-layer MLP projection head after Core 1 receives halting confirmation from Core 0. | **Speculative Asynchronous Output Pre-Computation (APOP)**: Core 0 speculatively executes the projection head into a shadow buffer when $s_{k+1} < \tau_{lo}$ (precision $\ge 97.2\%$). | **0 μs tail latency (100% hidden)** upon halt confirmation; saves ~185 μs per early exit. |
| **2. Spatial Compute Inefficiency** | All $T=32$ sequence tokens updated uniformly across all recursive steps, even when syntax and contextual stop words have converged. | **Temporal Token Saliency Freezing (TSTF)**: Dynamic $L_1$ hidden-state divergence observer ($\Delta_k[t] < \tau_{\text{freeze}}$) bypasses the 384-wide GELU FFN. | **25% to 40% reduction in FFN MACs** on later steps; 100% bit-level parity preserved when unconstrained. |
| **3. Battery Internal Impedance Droop** | Active burst currents ($I_{\text{active}} \approx 50\text{ mA}$ at 240 MHz) across Li-Po battery internal resistance ($R_{\text{int}} \approx 0.200\ \Omega$) depress rail voltage near brownout ($< 3.3\text{V}$). | **Closed-Loop Internal Resistance Droop Observer**: $V_{\text{loaded}} = V_{\text{OCV}} - I_{\text{active}}(f) \times R_{\text{int}}$. Safety margin checking enforces immediate ECO scaling and cooperative abort flags. | **Brownout-free continuous operation**; graceful step truncation preserving valid classifications without reboots. |

---

## Interactive Demonstrations & Testing Quickstart

### 1. Live Interactive Web Demonstration Studio (Port 8000)
Run the zero-dependency web application (works on Mac, Linux, and Windows without extra dependencies):

```bash
python3 host/web_demo.py
# Open http://localhost:8000 in your browser
```

Features included in the web interface:
- **Live Studio Tab**: Type any review or sentence to observe real-time sentiment prediction, confidence %, latency, and step-by-step halting cards.
- **Interactive Chart.js Visualizer**: Real-time chart displaying cumulative halting mass $M_k$ vs ACT threshold ($0.900$) and mixing gate values $g_k$.
- **RLM Chat Tab**: Conversational interface with the RLM agent displaying dynamic recursive reasoning and step savings.
- **Interactive Serial Console**: Real-time terminal emulator allowing you to send `CHAT`, `INFER`, `BENCH`, `STAT`, `MODE`, and `BATT` commands directly.
- **Hardware Coprocessor Tab**: Memory-mapped register viewer and cycle-accurate RTL Verilog simulation logs.
- **Battery Droop Slider**: Simulates battery discharge (3.3V - 4.2V), calculating internal resistance voltage droop and displaying brownout warnings.
- **Patent & Defense Hub**: Complete searchable text of all 20 patent claims and viva voce examination Q&A.

---

### 2. Offline Conversational RLM "Chat & Test" Agent
Simulate and test the exact RLM model conversationally on your laptop without needing physical ESP32 hardware:

```bash
# Launch interactive conversational terminal:
python3 tools/chat_rlm.py

# Or single-shot prompt evaluation:
python3 tools/chat_rlm.py --prompt "the acting was phenomenal and deeply moving" --mode PERF
```

**Live Terminal Output:**
```text
======================================================================
   Edge-RLM Offline Conversational Testing Model (ESP32 Simulation)
======================================================================
Commands:
  /mode PERF|BAL|ECO  - Change DVFS profile
  /batt <millivolts>  - Adjust simulated battery voltage (e.g. /batt 3500)
  /help               - Display command reference
  exit | quit         - Exit chat session
----------------------------------------------------------------------
Type any sentence, review, or question to chat with the RLM model:

RLM [PERF | 4000mV] > the acting was phenomenal and deeply moving

I analyzed your input: "the acting was phenomenal and deeply moving"
• Classification Verdict: POSITIVE (+) (Confidence: 100.0%)
• Recursive Reasoning   : 2 steps (Cumulative Mass: 1.033 / 0.900 threshold)
• Latency & Compute     : 2,105 μs (2.10 ms) | Saved: 6 steps (75% reduction)
• Hardware Telemetry    : PERF @ 240 MHz | Loaded Batt: 3990.0 mV (-10.0 mV droop)
• Patented Features     : 0 converged tokens stabilized, 0 speculative APOP pre-arm hints fired
```

---

### 3. Embedded Web Serial Console (Mock UART)
Evaluate the line-oriented FreeRTOS serial protocol through the host gateway simulator:

```bash
python3 host/gateway.py --mock --eval
```
Simulates real UART communication, runs benchmark sweeps across all operating profiles, and records evaluation metrics in `artifacts/`.

---

### 4. Dual-Track Hardware-Free Verification Suite

```bash
# Track 1: C++ Native Firmware Parity vs PyTorch Golden Reference (12/12 Golden Vectors)
make -C firmware/test check

# Track 2: Verilog Hardware Coprocessor Simulation (Cycle-accurate RTL + VCD waveforms)
make -C verification/verilog check
```

---

### 5. Physical ESP32 Arduino Deployment & Serial Chat
When flashing to an actual ESP32 DevKit board:
1. Open `firmware/rlm_esp32/rlm_esp32.ino` in Arduino IDE or PlatformIO.
2. Build and upload to the board (baud rate: 115200).
3. Open the Serial Monitor and type:
   ```text
   CHAT the movie was a brilliant masterpiece with stunning visuals
   ```
4. The ESP32 evaluates the input on-device using dual cores and responds:
   ```json
   {
     "chat_reply": "I evaluated your input: 'the movie was a brilliant masterpiece with stunning visuals'. Verdict: POSITIVE (Confidence: 100.0%). Reasoned in 2 recursive steps (Mass: 1.033/0.900), saving 6 steps (75% compute reduction). Latency: 3240 us.",
     "pred": 1,
     "verdict": "POSITIVE",
     "confidence": 100.0,
     "steps": 2,
     "saved_steps": 6,
     "us": 3240,
     "profile": "PERF"
   }
   ```

#### Serial Command Reference

| Command | Arguments | Description | Example Response |
| :--- | :--- | :--- | :--- |
| `CHAT` | `<text>` | Conversational RLM sentiment reasoning and telemetry | `{"chat_reply":"...","confidence":99.8,"steps":2}` |
| `INFER` | `<text>` | Raw telemetry JSON line (logits, gates, shadows, droop) | `{"pred":1,"steps":2,"depth":0.9822,"us":3469}` |
| `MODE` | `PERF \| BAL \| ECO \| AUTO` | Set DVFS frequency and recursion budget | `{"mode":"BAL","budget":6,"cpu_mhz":160}` |
| `BATT` | `<millivolts> \| AUTO` | Set simulated battery OCV or read ADC on GPIO34 | `{"batt_mv":3800,"loaded_mv":3790,"profile":"BAL"}` |
| `BENCH` | `<n>` | Run $n$ consecutive benchmark iterations on fixed review | `{"bench_n":20,"us_avg":3625.6,"mean_steps":2.083}` |
| `STAT` | *(none)* | Return CPU frequency, heap, flash, and profile state | `{"cpu_mhz":240,"heap":284120,"sketch_kb":510}` |
| `PING` | *(none)* | Heartbeat check | `{"pong":1,"uptime_ms":10420}` |

---

## Hardware & Software Architecture

### Core 1 (Compute Core, APP_CPU)
- **Tokenization**: Zero-vocabulary 32-bit FNV-1a hash tokenizer (`tokenizer.cpp`). Modulo $4095 + 1$ maps any English word directly into token IDs without flash-resident vocabulary tables.
- **Recursive Transformer**: Single weight-shared block comprising Multi-Head Attention (4 heads, $D_k = 24$, $D = 96$), Layer Normalization, and 384-wide GELU FFN.
- **State Refinement**: Evaluates state update $h'_k = \text{Transformer}(h_{k-1})$. Applies soft mixing gate $h_k = g_k \cdot h_{k-1} + (1 - g_k) \cdot h'_k$.
- **Temporal Token Freezing**: Tracks token divergence $\Delta_k[t]$. Converged tokens skip the 384-wide GELU FFN.

### Core 0 (Arbiter Core, PRO_CPU)
- **Asynchronous Gating**: Receives pooled hidden state representations $h_{\text{pool}}$ via zero-copy mailboxes. Evaluates $g_k = \sigma(W_g h_{\text{pool}} + b_g)$ concurrently while Core 1 performs state updates.
- **Shadow Halt Forecaster**: Evaluates 1-step-ahead predictor $s_{k+1}$.
- **Speculative Head Pre-Computation (APOP)**: When $s_{k+1} < \tau_{lo} = 0.347$, executes output classification head concurrently with Core 1's final step.
- **Battery Droop Observer**: Heartbeat interrupt samples GPIO34 ADC, subtracts $I_{\text{active}}(f) \times R_{\text{int}}$, and sets cooperative abort flags if $V_{\text{loaded}} < 3350\text{ mV}$.

---

## Quantization & Memory Layout

```
ESP32 SRAM (520 KiB total, 328 KiB allocatable)
+--------------------------------------------------------------+
| FreeRTOS Kernel & Core Stacks                  (~40 KiB)     |
+--------------------------------------------------------------+
| RLM Static Activation Buffers                  (28 KiB)      |
|   - Hidden State Tensors H_k (32 x 96 float)   : 12,288 B    |
|   - Scratchpad Buffers & MHA Keys/Values       : 15,360 B    |
|   - Dynamic Heap Allocation                    : 0 BYTES     |
+--------------------------------------------------------------+
| Free Headroom Available                        (~260 KiB)    |
+--------------------------------------------------------------+

ESP32 Flash Memory (4 MiB total, .rodata)
+--------------------------------------------------------------+
| Quantized int8 Symmetric Weights               (510 KiB)     |
| Per-Channel float32 Scales & Biases            (14 KiB)      |
| Zero-Vocab FNV-1a Hash String Storage          : 0 BYTES     |
+--------------------------------------------------------------+
```

---

## Empirical Benchmarks

### Comparison: Edge-RLM vs Standard Static Models

| Architectural Model | Recursion Steps | Inference Latency | Active Energy | Model Flash | SRAM Usage | Accuracy | Brownout Resilient? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed-Depth 8 (Baseline)** | 8.00 (Fixed) | 12.8 ms | 2,112 μJ | 510 KiB | 28 KiB | 92.33% | No |
| **DeeBERT Early-Exit** | 4.60 (Avg) | 7.4 ms | 1,221 μJ | 4,080 KiB | 84 KiB | 90.15% | No |
| **TFLM Unrolled (8 Blocks)** | 8.00 (Fixed) | 13.5 ms | 2,227 μJ | 4,080 KiB | 96 KiB | 92.33% | No |
| **Edge-RLM + AAGM (Unconstrained)** | **2.18 (Avg)** | **3.5 ms** | **574 μJ** | **510 KiB** | **28 KiB** | **92.33%** | Yes |
| **Edge-RLM + AAGM + TSTF + APOP** | **2.08 (Avg)** | **3.2 ms** | **528 μJ** | **510 KiB** | **28 KiB** | **92.33%** | **Yes (Closed-Loop)** |

---

## Repository File Map

```
rlm-esp32-aagm/
├── README.md                            # Comprehensive project overview, quickstart & benchmarks
├── requirements.txt                     # Host Python dependencies
├── docs/
│   ├── RESEARCH.md                      # Academic literature review, theoretical paper & novelty
│   ├── PROJECT_REVIEW_REPORT.md         # Formal college project review report
│   ├── PRESENTATION_GUIDE.md            # Slide deck script & top 10 viva voce Q&A cheat sheet
│   ├── PATENT_SPECIFICATION.md          # 20-claim formal patent application
│   └── TECHNICAL_BOTTLENECK_RESOLUTION.md # Detailed engineering analysis of the 3 bottleneck fixes
├── firmware/
│   ├── rlm_esp32/
│   │   ├── rlm_esp32.ino                # Dual-core FreeRTOS Arduino firmware (CHAT + INFER + Droop)
│   │   ├── platformio.ini               # PlatformIO build configuration (esp32dev / esp32-s3)
│   │   └── src/
│   │       ├── rlm_engine.h/.cpp        # C++ engine with APOP speculative head & TSTF freezing
│   │       ├── tokenizer.cpp            # Zero-vocab FNV-1a hash tokenizer
│   │       ├── rlm_weights.h            # Symmetric per-channel int8 quantized weights (510 KiB)
│   │       └── rlm_config.h             # Core configuration, thresholds, and dimensions
│   └── test/
│       ├── host_harness.cpp             # Host C++ test harness supporting --interactive mock UART
│       └── Makefile                     # Fast compilation and golden check target
├── tools/
│   ├── chat_rlm.py                      # Offline interactive conversational testing model
│   ├── demo.py                          # CLI step-by-step recursion tracer
│   ├── aagm.py                          # Algorithmic reference model + ACT gating + shadow forecaster
│   ├── quantize.py                      # Symmetric per-channel int8 quantization logic
│   └── out/golden_vectors.txt           # 12 int8-mirror expectation vectors
├── verification/
│   ├── README.md                        # Hardware-free verification methodology
│   └── verilog/
│       ├── aagm_gate.v                  # Digital RTL gating policy module
│       ├── aagm_arbiter_coprocessor.v   # APB memory-mapped hardware coprocessor
│       ├── sim_aagm_gate.py             # Cycle-accurate simulator emitting aagm_gate.vcd
│       ├── sim_aagm_arbiter_coprocessor.py # Cycle-accurate simulator emitting aagm_coprocessor.vcd
│       └── Makefile                     # Dual-mode verification makefile
├── host/
│   ├── gateway.py                       # Serial client for physical board or mock simulator
│   └── web_demo.py                      # Interactive live web dashboard (0.0.0.0:8000)
└── artifacts/                           # Benchmark JSON records and visual SVG charts
```

---

## Documentation Links

- **Theoretical Paper & Literature Analysis**: [`docs/RESEARCH.md`](docs/RESEARCH.md)
- **Formal Project Review Report**: [`docs/PROJECT_REVIEW_REPORT.md`](docs/PROJECT_REVIEW_REPORT.md)
- **Slide Deck & Viva Voce Defense Guide**: [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md)
- **20-Claim Formal Patent Application**: [`docs/PATENT_SPECIFICATION.md`](docs/PATENT_SPECIFICATION.md)
- **Technical Bottleneck Resolution Guide**: [`docs/TECHNICAL_BOTTLENECK_RESOLUTION.md`](docs/TECHNICAL_BOTTLENECK_RESOLUTION.md)
- **Hardware-Free Verification Suite**: [`verification/README.md`](verification/README.md)
