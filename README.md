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
   - [Focused Project Chat & Review Demo (Port 8000)](#1-focused-project-chat--review-demo-port-8000)
   - [Interactive 3D Code & Pipeline Graph](#2-interactive-3d-code--pipeline-graph)
   - [Offline RLM Sentiment Test CLI](#3-offline-rlm-sentiment-test-cli)
   - [Embedded Web Serial Console (Mock UART)](#4-embedded-web-serial-console-mock-uart)
   - [Dual-Track Hardware-Free Verification Suite](#5-dual-track-hardware-free-verification-suite)
   - [Physical ESP32 Arduino Deployment & Serial Chat](#6-physical-esp32-arduino-deployment--serial-chat)
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

### 1. Focused Project Chat & Review Demo (Port 8000)
Start the local app and open its streamlined chat interface:

```bash
python3 host/web_demo.py
# Open http://localhost:8000
```

Choose **Ask about the project** for source-backed answers from repository documents, or **Analyze a review** to run the actual Edge-RLM sentiment classifier and inspect its prediction, confidence, halting mass, recursive steps, and native host latency. The chat keeps recent turns for the current browser session. The legacy hardware dashboard remains available at [`/lab`](http://localhost:8000/lab), and the 3D graph at [`/graph`](http://localhost:8000/graph).

**Important model boundary:** the compact ESP32 RLM in this repository is a binary sentiment classifier; its weights are not a general text-generation model. Project chat checks for Ollama on loopback and uses it when available; otherwise it falls back to local document retrieval with file citations. Ollama is optional, free to run locally, and requires no API key. Prompts are not sent to a hosted AI API.

```bash
# Optional: install Ollama from https://ollama.com/download, start it, then fetch a model once:
ollama pull qwen2.5:3b
# With Ollama running, start the app in another terminal:
RLM_CHAT_BACKEND=ollama RLM_CHAT_MODEL=qwen2.5:3b python3 host/web_demo.py
```

If Ollama is unavailable, chat falls back to local repository search; sentiment analysis still runs through the existing C++ host harness. First analysis builds the harness automatically when needed.

```bash
python3 -m unittest discover -s host -p 'test_*.py'
```

---

### 2. Interactive 3D Code & Pipeline Graph

The same local server exposes a 3D, searchable view of the repository's Graphify code index. Start the server as above, then open **[http://localhost:8000/graph](http://localhost:8000/graph)** (or click **Graph** in the chat header).

```bash
python3 host/web_demo.py
# In another browser tab: http://localhost:8000/graph
```

The explorer includes:
- **Symbol graph** for functions, classes, rationale notes, and extracted code relationships.
- **File pipeline** view that rolls symbols and cross-file links up to their source files.
- Live search plus community, file, node-kind, and relationship filters; click a node to inspect neighbors and source location.
- Orbit, zoom, fit-to-view, and pause controls. Source files remain on this machine; the 3D renderer is loaded from the free jsDelivr CDN, so a network connection is needed for the renderer.

The page reads [`graphify-out/graph.json`](graphify-out/graph.json), the repository's [Graphify-Labs](https://github.com/Graphify-Labs) graph snapshot, directly—there is no database or Python package to install. To reflect code changes, refresh/regenerate that Graphify index using the local Graphify workflow; the page fetches the current JSON each time it loads. Rendering uses the open-source [3d-force-graph](https://github.com/vasturiano/3d-force-graph) library.

---

### 3. Offline RLM Sentiment Test CLI
Run the parity-matched host inference harness without physical ESP32 hardware (this tests sentiment classification; project Q&A is in the web chat above):

```bash
# Launch interactive sentiment test shell:
python3 tools/chat_rlm.py

# Or single-shot prompt evaluation:
python3 tools/chat_rlm.py --prompt "the acting was phenomenal and deeply moving" --mode PERF
```

**Sample inference output:**
```text
Input: "the acting was phenomenal and deeply moving"
Prediction: POSITIVE (100.0% confidence)
Adaptive depth: 2 steps; halting mass 1.033 / 0.900
Host inference: 2,081 μs at 240 MHz (PERF)
Battery estimate: 3990.0 mV loaded, 10.0 mV droop
Compared with the 8-step budget: 6 steps skipped. This classifier predicts review sentiment; it does not generate text.
```

---

### 4. Embedded Web Serial Console (Mock UART)
Evaluate the line-oriented FreeRTOS serial protocol through the host gateway simulator:

```bash
python3 host/gateway.py --mock --eval
```
Simulates real UART communication, runs benchmark sweeps across all operating profiles, and records evaluation metrics in `artifacts/`.

---

### 5. Dual-Track Hardware-Free Verification Suite

```bash
# Track 1: C++ Native Firmware Parity vs PyTorch Golden Reference (12/12 Golden Vectors)
make -C firmware/test check

# Track 2: Verilog Hardware Coprocessor Simulation (Cycle-accurate RTL + VCD waveforms)
make -C verification/verilog check
```

---

### 6. Physical ESP32 Arduino Deployment & Serial Chat
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
| `CHAT` | `<text>` | Binary sentiment classification and recursive-inference telemetry | `{"chat_reply":"...","confidence":99.8,"steps":2}` |
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
│   ├── chat_rlm.py                      # Offline interactive sentiment test shell
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
├── graphify-out/
│   ├── graph.json                       # Local Graphify symbol and relationship index
│   └── manifest.json                   # Indexed-source manifest
├── host/
│   ├── chat.html                        # Focused chat UI: project Q&A or actual sentiment inference
│   ├── chat_agent.py                    # Local source retrieval + optional loopback Ollama + RLM adapter
│   ├── test_chat_agent.py               # Chat retrieval, local backend and classifier tests
│   ├── gateway.py                       # Serial client for physical board or mock simulator
│   ├── pipeline_graph.html              # Searchable interactive 3D code/pipeline explorer
│   └── web_demo.py                      # Local chat, legacy /lab, graph and API server (0.0.0.0:8000)
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
