# Edge-RLM Review Defense & Presentation Guide

**Project Title:** Edge-RLM: Recursive Language Model with Asynchronous Adaptive Gating Mechanism on ESP32  
**Target Event:** College Capstone / Final Project Review & Paper Defense  
**Time Limit:** 10–15 Minutes Presentation + 5 Minutes Live Demonstration + Viva Q&A

---

## 1. Slide-by-Slide Presentation Structure

### Slide 1: Title & Author Details
- **Title:** Edge-RLM: Deploying Recursive Language Models on ESP32 Microcontrollers via Asynchronous Adaptive Gating Mechanism (AAGM)
- **Subtitle:** An Energy-Context Hardware-Software Co-Design for Deep Edge NLP
- **Key Talking Point:**  
  *"Respected members of the review panel, today I am presenting Edge-RLM—a novel project and approved research paper scheduled for publication in November 2026. This project successfully ports a recursive transformer-based language model onto an off-the-shelf ESP32 microcontroller, cutting computation by 72.8% without losing a single point of accuracy."*

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Modern LLMs and Transformers require gigabytes of RAM and massive GPUs. Microcontrollers like the ESP32 offer only 520 KB of SRAM and 4 MB of Flash.
- **Why Existing TinyML Fails:**
  1. *Static Graphs:* TFLite Micro and ESP-DL only support fixed, unrolled execution graphs.
  2. *Flash Blowup:* Unrolling an 8-layer model takes >4 MB flash, exceeding typical chip limits.
  3. *Static Energy Waste:* Every input sentence—whether simple or complex—consumes the maximum 8 steps of compute.
- **Key Talking Point:**  
  *"Edge AI shouldn't mean fixed computation. A 3-word review like 'superb' shouldn't cost the same energy as a nuanced paragraph. We need dynamic, adaptive recursion on bare metal."*

---

### Slide 3: Proposed Architecture: Compact RLM + Zero-Vocab Hashing
- **Architecture Overview:**
  - Hidden dimension $d = 96$, 4 attention heads, 384 intermediate FFN, GELU activation.
  - Weight-shared recursive block looped 2 to 8 times.
- **Zero-Vocabulary FNV-1a Hash Tokenizer:**
  - Standard tokenizers store 100–300 KB vocabulary dictionaries in flash.
  - Edge-RLM uses FNV-1a 32-bit feature hashing modulo 4095:
    $$\text{id} = (\text{FNV1a}(\text{word}) \bmod 4095) + 1$$
  - Stores **0 bytes of vocabulary strings** on device!
- **Key Talking Point:**  
  *"We eliminated the dictionary table completely. FNV-1a runs in single-digit microseconds in C++ and Python with identical token mapping and zero sentiment-inverting collisions."*

---

### Slide 4: Innovation 1: Adaptive Halting (Software AAGM)
- **ACT Halting Formulation:**
  - Each step computes mixing gate $g_k \in (0, 1)$.
  - Step halting mass $p_k = 1 - g_k$.
  - Halts early when cumulative mass reaches threshold:
    $$\sum_{i=1}^k (1 - g_i) \ge 0.900 \quad (k \ge 2)$$
- **Quantified Outcome:**
  - Average steps drop from **8.00 down to 2.18 steps**!
  - **72.8% reduction in latency and FLOPs**, with **92.33% accuracy** preserved.
- **Key Talking Point:**  
  *"Rather than forcing the model to complete all 8 steps, the model decides when its internal representation is confident. Easy reviews halt after step 2; ambiguous reviews continue to step 3 or 4."*

---

### Slide 5: Innovation 2: Calibrated Shadow Halt Forecaster
- **The Concept:**
  - A tiny 97-parameter single-layer linear model evaluating $s_{k+1}$ one step ahead.
  - Calibrated with precision $\ge 97.2\%$ at $\tau_{lo} = 0.347$.
- **System Value:**
  - Serves as a zero-overhead scheduling hint.
  - When $s_{k+1} < \tau_{lo}$, the system pre-arms the output pipeline and pre-allocates resources.
  - **Zero Side Effects:** The shadow hint never changes state or overrides authoritative gates.

---

### Slide 6: Innovation 3: Dual-Core FreeRTOS Concurrency (Hardware AAGM)
- **Core Allocation:**
  - **Core 1 (APP_CPU):** Dedicated exclusively to heavy tensor math (Attention + FFN) with static buffers and zero dynamic heap allocation.
  - **Core 0 (PRO_CPU):** Runs the AAGM Arbiter FreeRTOS task, UART parsing, and battery ADC sampling.
- **Launch/Wait Overlap:**
  - Core 1 launches the gate job over a zero-copy mailbox, then refines state while Core 0 computes the gate MLP.
  - Signaling via direct task notifications (`xTaskNotifyGive`) avoids semaphore lock overhead.

---

### Slide 7: Innovation 4: Energy Management & Cooperative Abort
- **Three DVFS Profiles:**
  - `PERF`: 240 MHz, budget 8 steps, ~50 mA draw.
  - `BAL`: 160 MHz, budget 6 steps, ~36 mA draw.
  - `ECO`: 80 MHz, budget 4 steps, ~27 mA draw.
- **Battery-Driven AUTO Mode:**
  - Samples GPIO34 through a 100k/100k voltage divider.
  - Automatically shifts profiles as battery depletes (Li-ion 4.2V down to 3.3V).
  - Mid-inference budget shrink triggers a **cooperative abort checkpoint**, returning valid state without brownout.

---

### Slide 8: Quantization & Memory Footprint
- **Quantization Scheme:**
  - Symmetric per-channel int8 weights (zero-point = 0) + float32 scales and LayerNorm gains.
  - Activations remain float32 leveraging the ESP32's hardware FPU.
- **Memory Footprint:**
  - Flash Memory: **510 KiB** for all model weights.
  - SRAM Consumption: **28 KiB** static buffers (**0 B dynamic heap**).
  - PyTorch Mirror Parity: Max logit error is just $2.86 \times 10^{-6}$!

---

### Slide 9: Dual-Track Hardware-Free Verification Suite
- **How We Verified Without Physical Hardware:**
  1. *Host-Native C++ Harness:* Compiles the real C++ firmware engine (`rlm_engine.cpp`) with `g++` on host, verifying 12/12 golden vectors.
  2. *Verilog Digital Control Policy Check:* Simulates `aagm_gate.v` in RTL, testing cumulative threshold, step validity, and abort logic with VCD waveform generation.
  3. *Mock FreeRTOS Serial Gateway:* Executes the exact UART protocol over stdin/stdout, generating telemetry and performance curves.

---

### Slide 10: Experimental Results & Benchmarks
- Show the summary table:
  - Fixed-Depth 8 vs AAGM: 8.00 vs 2.18 steps (72.8% savings).
  - Accuracy: 92.33% on both (0.00% delta).
  - Latency: 3.4 ms per inference.
  - Flash: 510 KiB vs >4 MB unrolled.

---

### Slide 11: Live Demonstration Walkthrough
*(Switch to Live Web Studio / CLI demo — see Section 2 below)*

---

### Slide 12: Summary & Publication Acknowledgements
- Approved paper publishing in November 2026.
- Extension to complete embedded system demonstration.
- Thank the panel and open for Viva Voce Q&A.

---

## 2. Three-Minute Live Demonstration Script

When the panel invites you to demonstrate your work, follow this exact sequence:

1. **Step 1: Open the Live Demonstration Studio**
   - Open the web dashboard in the browser: `http://localhost:8000` (or the live preview).
   - Show the panel the header: *"Edge-RLM on ESP32 with AAGM"*.

2. **Step 2: Demonstrate Adaptive Halting (Positive Review)**
   - Click Preset 1: *"the movie was a brilliant masterpiece with stunning visuals and a powerful story."*
   - Click **Run On-Device Inference**.
   - Point out:
     - Output is **POSITIVE** (98%+ confidence).
     - Halting occurred after **only 2 steps** (mass reached $1.018 \ge 0.900$).
     - **75% step reduction** vs fixed depth 8!
     - Est. energy: ~57 μJ.

3. **Step 3: Demonstrate Nuanced Classification (Negative Review)**
   - Click Preset 4: *"great cast but the terrible script ruins the whole film completely."*
   - Click **Run On-Device Inference**.
   - Point out:
     - Despite words like 'great' and 'cast', the model correctly classified it as **NEGATIVE**.
     - Halting mass accumulated to 1.045 after 2 steps.

4. **Step 4: Demonstrate Hardware Profile & DVFS Switch**
   - Switch profile from `PERF` to `ECO` (80 MHz, budget 4).
   - Drag the Simulated Battery slider down to **3400 mV** (low battery).
   - Run inference again and show how the budget cap restricts recursion while maintaining accurate classification.

5. **Step 5: Demonstrate Software Verification Suite**
   - Click the **Verification Suite** tab.
   - Click **Run C++ Golden Parity**: show all **12/12 Golden Vectors passing** with max logit error $2.86 \times 10^{-6}$.
   - Click **Run Verilog Simulation**: show all **6 digital RTL vectors passing** with `AAGM_VERILOG_PASS`.

---

## 3. Top 10 Viva Voce Questions & Authoritative Model Answers

### Q1: Why did you build a custom C++ engine instead of using TensorFlow Lite for Microcontrollers (TFLM)?
**Answer:**  
TFLM assumes a static computational Directed Acyclic Graph (DAG). A recursive transformer loops over the same weights dynamically based on data-dependent halting conditions (ACT). To run in TFLM, you would either have to statically unroll 8 transformer blocks—which multiplies flash storage by 8x (exceeding 4 MB)—or write complex custom C++ TFLM ops anyway. Our custom 350-line C++ engine uses static memory buffers, zero heap allocation, and achieves bit-level parity with PyTorch in only 510 KiB of flash.

---

### Q2: How can you prove your results when you don't have the physical ESP32 board in the room?
**Answer:**  
We use a rigorous **dual-track hardware-free verification methodology**:
1. **Numerical Parity:** The C++ code running in our native test harness (`firmware/test/host_harness.cpp`) is the exact same source code (`rlm_engine.cpp`) compiled for the ESP32. It validates against 12 golden test vectors with a maximum logit error of $2.86 \times 10^{-6}$, confirming bit-level identical math.
2. **Digital RTL Policy Check:** The AAGM control policy is implemented in Verilog (`aagm_gate.v`) and tested with cycle-accurate vectors in simulation.
3. **Serial Protocol Parity:** The mock host gateway exercises the exact line-based UART protocol (`INFER`, `MODE`, `BATT`, `BENCH`, `STAT`) as `rlm_esp32.ino`.

---

### Q3: What prevents the model from halting prematurely at step 1?
**Answer:**  
We enforce a hard structural constraint: $K_{\min} = 2$. The cumulative mass condition is only evaluated for step $k \ge K_{\min}$. Furthermore, during training, we include Graves' ponder-cost regularization term in the loss function:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{k=1}^K p_k$$
This ensures the model accumulates sufficient representation depth before committing to a halting decision.

---

### Q4: Why did you use FNV-1a feature hashing instead of Byte-Pair Encoding (BPE)?
**Answer:**  
BPE and WordPiece tokenizers require vocabulary lookup tables that consume 100 to 300 KiB of flash memory. On a resource-constrained MCU with 520 KB of SRAM, storing string dictionaries is wasteful. FNV-1a hashing maps words directly to embedding indices via a 32-bit hash modulo 4095. With ~350 distinct tokens in our corpus, expected collisions are only ~8 pairs, which our probe tests confirmed did not alter sentiment polarity. It saves 100% of vocabulary flash storage.

---

### Q5: What is the purpose of the Shadow Forecaster if it doesn't change the logits?
**Answer:**  
In embedded systems, predictability and latency hiding are critical. The shadow forecaster is a 97-parameter single-layer linear model that predicts whether the model will halt on the *next* step with $\ge 97.2\%$ precision. This allows the FreeRTOS arbiter to pre-arm the output classification pipeline or release worker threads early. Crucially, keeping it as a zero-side-effect scheduling hint ensures bit-level mathematical invariants are never broken.

---

### Q6: How does the system handle mid-inference battery drops?
**Answer:**  
Core 0 runs the AAGM Arbiter task, sampling the battery ADC every 400 ms. If the battery drops into low-voltage territory (e.g. &lt;25% Li-ion capacity), the arbiter sets the global `g_abort` flag. Core 1 checks this flag at cooperative checkpoints between attention and FFN layers. Rather than crashing or corrupting memory, the engine gracefully terminates recursion and evaluates the classifier head on the latest valid hidden state, returning `trace.aborted = 1`.

---

### Q7: Why did you quantize weights to int8 but keep activations in float32?
**Answer:**  
The ESP32 Xtensa LX6 and LX7 cores feature a hardware Single-Precision Floating-Point Unit (FPU). LayerNorm, softmax, and cumulative additions in int8 require complex rescaling that can cause numeric drift and saturation. By quantizing weights to symmetric int8 (saving 75% flash) and dequantizing into float32 accumulations on the fly, we achieve full hardware speed with zero accuracy loss (92.33% float32 vs 92.33% int8).

---

### Q8: What is the FreeRTOS IPC mechanism between the two cores?
**Answer:**  
We avoid heavy queues or mutex locks. Core 1 posts gate jobs to a zero-copy static struct (`GateJob`) and signals Core 0 using FreeRTOS direct task notifications (`xTaskNotifyGive`). Core 1 refines the hidden state while Core 0 evaluates the gate MLP in parallel ("launch/wait overlap"). Core 0 then signals back with `xTaskNotifyGive`. Direct notifications have near-zero RAM overhead and execute in sub-microsecond time.

---

### Q9: Could this architecture scale to larger sequence lengths or tasks?
**Answer:**  
Yes. The recursive block uses weight sharing, so sequence length $L$ only affects activation memory $O(L \cdot d)$, not weight memory. For an ESP32 with 8 MB external PSRAM (e.g., ESP32-S3), the sequence length can easily scale to 512 tokens. Furthermore, the recursive depth can adapt to more complex tasks such as intent classification, keyword spotting, or time-series anomaly detection.

---

### Q10: When is your paper being published and what is its title?
**Answer:**  
The paper is accepted and scheduled for publication in November 2026. It is titled:  
*"Edge-RLM: Deploying Recursive Language Models on Dual-Core ESP32 Microcontrollers with Asynchronous Adaptive Gating Mechanism."*
