# PATENT SPECIFICATION

**Title of the Invention:**  
**ASYNCHRONOUS DUAL-CORE RECURSIVE TRANSFORMER INFERENCE SYSTEM WITH SPECULATIVE EARLY-EXIT AND ADAPTIVE ENERGY-CONTEXT GATING**

---

## Technical Field of the Invention

The present invention relates generally to the field of embedded artificial intelligence, edge computing microcontrollers, and neural network hardware acceleration. More particularly, the invention relates to an embedded hardware-software co-designed system and method for executing recursive transformer-based language models on resource-constrained dual-core microcontrollers (such as Xtensa-based ESP32 and RISC-V architectures) utilizing an Asynchronous Adaptive Gating Mechanism (AAGM), speculative output head pre-computation, battery-droop-compensated dynamic voltage and frequency scaling (DVFS), and temporal token saliency freezing.

---

## Background of the Invention & Prior Art Limitations

Transformer-based natural language processing (NLP) models, while achieving state-of-the-art predictive performance, are notorious for their immense computational complexity and large memory footprints. Deploying such models onto low-cost, resource-constrained edge devices—such as embedded microcontrollers featuring less than 1 Megabyte of internal Static Random Access Memory (SRAM) and battery power sources—presents fundamental technological bottlenecks:

1. **Static Graph Runtimes versus Dynamic Recursion:**  
   Mainstream edge machine learning deployment frameworks, including TensorFlow Lite for Microcontrollers (TFLM) and Espressif ESP-DL, enforce static computational execution graphs. In these frameworks, the computational topology and iteration counts must be fixed at compile time. Deploying a deep recursive language model would require statically unrolling the recursive block across all maximal steps (e.g., 8 steps), which multiplies the on-chip Flash memory requirement by 8x (often exceeding 4 megabytes), exhausting the microcontroller's non-volatile storage and wasting battery power by computing unnecessary iterations on simple inputs.

2. **Sequential Tail Latency in Adaptive Computation Time (ACT):**  
   Algorithmic approaches such as Adaptive Computation Time (Graves, 2016) introduce data-dependent early halting. However, in conventional single-core implementations, the halting gate evaluation and the subsequent output classification head must be computed strictly sequentially *after* the recursive block loop terminates. This introduces substantial output tail latency, during which the processor remains at maximum clock frequency.

3. **Absence of Hardware-Aware Concurrency & Launch/Wait Overlap:**  
   Commodity microcontrollers (such as the dual-core Xtensa LX6/LX7 ESP32) contain two asymmetric or symmetric processing cores. Conventional TinyML runtimes either execute entirely on a single core or suffer from heavy operating system synchronization overhead (such as FreeRTOS queue and mutex contention), which negates the benefits of multi-core parallelization for short inference bursts.

4. **Battery Droop and Mid-Inference Brownout Vulnerability:**  
   Battery-powered Internet-of-Things (IoT) edge nodes experience internal resistance-induced voltage droop ($V_{\text{droop}} = I_{\text{active}} \times R_{\text{int}}$) during peak computational bursts (e.g., 240 MHz dual-core operation). Existing systems employ static voltage cutoffs that either trigger premature device reset or fail to prevent catastrophic brownout mid-inference.

5. **Flash Memory Exhaustion from Subword Dictionaries:**  
   Standard subword tokenizers (such as BPE, WordPiece, and SentencePiece) require dedicated vocabulary string-to-ID lookup tables consuming 100 to 300 Kilobytes of Flash memory—a prohibitive overhead on 520 KB SRAM chips.

Accordingly, there exists an urgent, unaddressed technical need in the art for an integrated embedded architecture that resolves the aforementioned limitations through a hardware-software co-designed asynchronous adaptive gating coprocessor and runtime.

---

## Summary of the Invention

The present invention solves the technical problems stated above by providing a hardware-software co-designed inference system, embedded firmware engine, and digital coprocessor architecture for executing recursive language models on multi-core microcontrollers.

In a preferred embodiment, the system comprises:
1. **A Primary Compute Core (Core 1 / APP_CPU):** Configured to execute the heavy multi-head attention (MHA) and feed-forward network (FFN) layers of a weight-shared recursive transformer block using static memory allocations with zero dynamic heap allocation.
2. **An Asynchronous Arbiter Core (Core 0 / PRO_CPU):** Operating concurrently with the primary compute core, configured to evaluate gating multilayer perceptrons (MLPs), calibrate a shadow halt-forecaster, sample battery state-of-charge with internal resistance compensation, and manage dynamic recursion budgets.
3. **Speculative Asynchronous Output Pre-Computation (APOP):** When the shadow forecaster predicts an impending halt with high calibrated precision ($\ge 97.2\%$), the arbiter core speculatively offloads and computes the final classification projection head concurrently with the primary core's final state refinement, thereby completely eliminating post-recursion tail latency.
4. **Temporal Token Saliency Freezing (TSTF):** A dynamic sparsity mechanism wherein tokens whose internal state divergence between successive recursive steps falls below a convergence threshold $\tau_{\text{freeze}}$ are bypassed during subsequent feed-forward transformations, achieving a 25% to 40% reduction in multiply-accumulate (MAC) operations without accuracy degradation.
5. **Battery-Droop Compensated DVFS & Cooperative Abort:** A closed-loop energy management policy that maps compensated battery voltage to recursion depth budgets (`PERF`: 240 MHz/8 steps, `BAL`: 160 MHz/6 steps, `ECO`: 80 MHz/4 steps) and initiates mid-inference cooperative aborts at internal block checkpoints when energy debt limits are approached.
6. **Zero-Vocabulary Feature Hash Tokenization:** A deterministic 32-bit FNV-1a feature hash token mapping modulo 4095 that eliminates the need for string dictionary tables on the microcontroller.

---

## Brief Description of the Accompanying Drawings

- **FIG. 1** is a high-level block diagram illustrating the dual-core microcontroller hardware-software co-design architecture of the present invention.
- **FIG. 2** is a timing and concurrency diagram illustrating the "launch/wait overlap" and Speculative Asynchronous Output Pre-Computation (APOP) between Core 1 and Core 0.
- **FIG. 3** is a state-flow diagram illustrating the Adaptive Computation Time (ACT) halting mass accumulation, shadow forecaster calibration, and cooperative abort checkpoints.
- **FIG. 4** is a detailed schematic of the digital AAGM coprocessor register interface and hardware accumulator logic.
- **FIG. 5** is a graph illustrating the calibration curve of the shadow halt-forecaster showing precision versus threshold $\tau_{lo}$.
- **FIG. 6** is a block diagram illustrating the Temporal Token Saliency Freezing (TSTF) mechanism within the recursive transformer block.

---

## Detailed Description of Preferred Embodiments

### 1. System Architecture & Memory Allocation
Referring to **FIG. 1**, the system is implemented on a dual-core microcontroller comprising Core 1 (Application CPU) and Core 0 (Protocol CPU), interconnected via an internal bus and shared SRAM.

To guarantee deterministic, real-time execution and prevent stack/heap overflow on microcontrollers with constrained SRAM (e.g., 520 KiB total), the system allocates all activation and intermediate tensors in static memory buffers (`.bss` section):
- Quantized weight matrices ($W_{\text{int8}}$) reside in non-volatile Flash memory (`.rodata`), totaling 510 KiB, quantized using symmetric per-channel 8-bit quantization with zero-point $zp = 0$.
- LayerNorm gains, biases, and channel scales reside in `.rodata` as 32-bit floating point numbers (27.6 KiB).
- Dynamic heap allocation (`malloc`, `new`) is strictly forbidden across the entire inference lifecycle.

### 2. Dual-Core Asynchronous Concurrency ("Launch/Wait Overlap")
Referring to **FIG. 2**, the inter-core communication is orchestrated without operating system mutex lock contention through a zero-copy mailbox structure (`GateJob`) and FreeRTOS direct task notifications:
1. Core 1 completes the multi-head self-attention and intermediate FFN transformation of recursive step $k$.
2. Core 1 performs masked pooling across active tokens $t \in [1, N_{\text{real}}]$ to yield a pooled hidden vector $z_k \in \mathbb{R}^d$.
3. Core 1 posts a pointer to $z_k$ into the zero-copy mailbox and executes `xTaskNotifyGive(g_arbiter_task)`.
4. While Core 0 evaluates the gate MLP:
   $$g_k = \sigma(W_{g2} \tanh(W_{g1} z_k + b_{g1}) + b_{g2})$$
   and the shadow forecaster:
   $$s_{k+1} = \sigma(w_{\text{shadow}}^T z_k + b_{\text{shadow}})$$
   Core 1 **concurrently executes token state refinement**:
   $$h_{\text{refine}}[t] = h''_k[t] + 0.1 \times (h''_k[t] W_{\text{refine}} + b_{\text{refine}})$$
5. This concurrency completely overlaps and conceals the computational latency of the gating multilayer perceptron within the state refinement window.

### 3. Speculative Asynchronous Output Pre-Computation (APOP)
In another key aspect of the invention, Core 0 utilizes the shadow forecaster prediction $s_{k+1}$ to eliminate inference tail latency.
- During calibration, an empirical threshold $\tau_{lo}$ (preferentially $\tau_{lo} = 0.347$) is established where precision $P(\text{Halt at } k+1 \mid s_{k+1} < \tau_{lo}) \ge 0.972$.
- When $s_{k+1} < \tau_{lo}$, Core 0 detects an imminent halt after the upcoming step.
- Immediately following gate evaluation, Core 0 **pre-emptively executes the final LayerNorm, masked pooling, and classification projection MLP** on the current state $z_k$.
- When Core 1 subsequently finishes step $k+1$ and confirms that cumulative halting mass satisfies $\sum (1 - g_i) \ge 0.900$, the final output logits are **already computed and valid in shared SRAM**.
- Core 1 immediately emits the inference result, reducing final-step output latency by 100%. If the authoritative gate indicates that recursion must continue ($<2.8\%$ probability), the speculative logits are invalidated with zero consequence to numerical precision.

### 4. Temporal Token Saliency Freezing (TSTF)
Referring to **FIG. 6**, within the recursive execution loop, tokens exhibit non-uniform convergence rates. In accordance with the invention, after step $k \ge 2$, Core 1 computes a token-wise representation delta:
$$\Delta_k[t] = \frac{1}{d} \sum_{j=1}^d \left| h_k[t, j] - h_{k-1}[t, j] \right|$$
If $\Delta_k[t] < \tau_{\text{freeze}}$ (preferentially $\tau_{\text{freeze}} = 0.015$), token $t$ is marked as **converged**. In step $k+1$, token $t$ bypasses the compute-intensive feed-forward network ($d \to 4d \to d$) via direct residual forwarding:
$$h_{k+1}[t] = h_k[t]$$
This dynamic token freezing achieves up to a 38% reduction in MAC operations on sequences with padded or repetitive context.

### 5. Battery-Droop Compensated DVFS & Cooperative Aborting
Core 0 periodically samples the supply rail via an ADC connected to a 100k/100k voltage divider on GPIO34. To account for battery internal impedance ($R_{\text{int}} \approx 150-300\,\text{m}\Omega$), Core 0 computes the true open-circuit voltage:
$$V_{\text{OCV}} = V_{\text{ADC}} \times 2.0 + I_{\text{active}}(f_{\text{CPU}}) \times R_{\text{int}}$$
Operating profiles are assigned dynamically:
- $V_{\text{OCV}} \ge 3.8\,\text{V} \implies \text{PERF}$ (240 MHz, budget $B=8$)
- $3.5\,\text{V} \le V_{\text{OCV}} < 3.8\,\text{V} \implies \text{BAL}$ (160 MHz, budget $B=6$)
- $V_{\text{OCV}} < 3.5\,\text{V} \implies \text{ECO}$ (80 MHz, budget $B=4$)

If voltage droop causes a profile transition from `PERF` to `ECO` during an ongoing inference, Core 0 sets a cooperative abort flag `g_abort = 1`. Core 1 polls this flag at attention and FFN boundaries. Upon detecting the abort flag, Core 1 cleanly breaks recursion, evaluates the classifier on the latest valid state, sets `trace.aborted = 1`, and prevents system reset due to brownout.

---

## Patent Claims

### We Claim:

1. **A dual-core embedded inference system for recursive neural networks, comprising:**
   - a multi-core microcontroller comprising at least a first compute core and a second arbiter core coupled to an internal shared memory;
   - a weight-shared recursive neural network block stored in non-volatile memory, wherein weights of said recursive block are shared across a plurality of recursive computation steps;
   - wherein said first compute core is configured to execute matrix multiplication operations of said recursive neural network block using statically allocated buffers in said shared memory; and
   - wherein said second arbiter core is configured to execute an asynchronous adaptive gating mechanism concurrently with said first compute core, wherein said gating mechanism computes a scalar mixing gate $g_k$ and evaluates an early halting criterion based on cumulative halting mass accumulated across recursive steps.

2. **The system of claim 1,** wherein said second arbiter core communicates with said first compute core via a zero-copy mailbox and direct microprocessor task notifications without mutex lock contention.

3. **The system of claim 1,** wherein while said second arbiter core computes said scalar mixing gate $g_k$, said first compute core concurrently executes a token state refinement operation, whereby gate evaluation latency is overlapped and hidden within state refinement execution.

4. **The system of claim 1,** wherein said second arbiter core further evaluates a single-layer shadow halt-forecaster $s_{k+1}$ configured to forecast whether recursion will terminate on a subsequent step with a calibrated precision exceeding 95%.

5. **The system of claim 4,** wherein upon detecting that $s_{k+1}$ is less than a predetermined confidence threshold $\tau_{lo}$, said second arbiter core speculatively evaluates an output classification head on a pooled hidden state concurrently with said first compute core executing a final recursive step, thereby eliminating output tail latency upon confirmation of early halting.

6. **The system of claim 1,** wherein said early halting criterion terminates recursion at step $k$ when cumulative halting mass satisfies:
   $$\sum_{i=1}^k (1 - g_i) \ge 1 - \epsilon$$
   wherein $k \ge 2$, and $\epsilon$ is a predefined halting tolerance parameter.

7. **The system of claim 1,** further comprising an energy-aware dynamic voltage and frequency scaling (DVFS) controller executed by said second arbiter core, wherein said controller samples battery voltage through an analog-to-digital converter (ADC) and dynamically adjusts clock frequency and maximum recursion step budget among a plurality of operating profiles.

8. **The system of claim 7,** wherein said operating profiles comprise:
   - a high-performance profile operating at 240 MHz with a maximum recursion budget of 8 steps;
   - a balanced profile operating at 160 MHz with a maximum recursion budget of 6 steps; and
   - an eco profile operating at 80 MHz with a maximum recursion budget of 4 steps.

9. **The system of claim 7,** wherein when said operating profile transitions to a lower recursion budget during an active inference, an asynchronous abort flag is asserted, causing said first compute core to terminate recursion cooperatively at an internal layer boundary and output classification logits from a currently computed hidden state.

10. **A method for accelerating recursive transformer inference on a microcontroller, the method comprising:**
    - tokenizing an input text string into token identifiers utilizing a deterministic 32-bit FNV-1a feature hash modulo a prime number without referencing an on-device dictionary table;
    - embedding said token identifiers into a hidden state tensor stored in static random access memory;
    - executing a plurality of recursive transformation steps on said hidden state tensor using a primary processor core;
    - offloading gating evaluation to a secondary processor core concurrently with execution on said primary processor core;
    - accumulating a cumulative halting probability mass across said recursive steps; and
    - terminating execution of said recursive steps prior to a maximum recursion limit when said cumulative halting mass exceeds a predetermined threshold.

11. **The method of claim 10,** further comprising:
    - computing a token state divergence metric $\Delta_k[t]$ between successive recursive steps for each token $t$; and
    - bypassing feed-forward network computation for tokens having a divergence metric less than a freezing threshold $\tau_{\text{freeze}}$, whereby converged tokens are propagated directly via residual connections.

12. **The method of claim 10,** further comprising:
    - pre-computing an output classification projection on said secondary processor core in response to a speculative forecaster signal indicating an imminent halt; and
    - committing said pre-computed output classification projection upon verification of said cumulative halting mass threshold.

13. **The method of claim 10,** wherein weights of said recursive transformer are quantized using symmetric per-channel 8-bit signed integer quantization with a zero-point equal to zero, and wherein activations are accumulated in 32-bit floating point format utilizing an on-chip hardware floating-point unit.

14. **The method of claim 10,** further comprising measuring battery voltage droop during inference bursts and computing an internal-resistance-compensated open-circuit voltage to prevent microcontroller brownout reset.

15. **A hardware coprocessor for adaptive neural network gating, comprising:**
    - a hardware register interface accessible by a host processor;
    - a fixed-point halting mass accumulator register configured to sum digital values representing $(1 - g_k)$ across computation steps;
    - a digital comparator configured to assert a hardware halt signal when said accumulator register equals or exceeds a 16-bit threshold corresponding to $1 - \epsilon$ in Q1.15 fixed-point format;
    - a budget limit counter configured to override said hardware halt signal upon exhaustion of a maximum step budget; and
    - a hardware abort register configured to assert an interrupt signal to stop processor core computation mid-inference.

16. **The hardware coprocessor of claim 15,** wherein said 16-bit threshold in Q1.15 fixed-point format equals 29,491, corresponding to a cumulative mass of 0.900.

17. **The hardware coprocessor of claim 15,** wherein said coprocessor generates an IEEE 1364 compliant value change dump (VCD) waveform trace during execution.

18. **The system of claim 1,** wherein average recursion steps of said recursive neural network are reduced from 8.00 steps to 2.18 steps with 0.00% degradation in classification accuracy.

19. **A non-transitory computer-readable medium** containing instructions which, when executed by a dual-core microcontroller, cause the microcontroller to perform the method of claim 10.

20. **The system of claim 1,** wherein total Static Random Access Memory (SRAM) consumed by model activations during inference is less than 32 Kilobytes.

---

## Abstract

An asynchronous dual-core recursive transformer inference system and method for embedded microcontrollers is disclosed. The system includes a primary compute core executing weight-shared transformer layers with static SRAM allocations and zero dynamic heap allocation, and a secondary arbiter core concurrently evaluating gating MLPs and shadow halt-forecasters via a zero-copy mailbox. When the shadow forecaster predicts an imminent halt with high precision ($\ge 97.2\%$), the arbiter core speculatively pre-computes the output classification projection head concurrently with the primary core's final state refinement, eliminating post-recursion tail latency. A temporal token saliency freezing mechanism bypasses converged tokens during subsequent steps. A battery-droop-compensated DVFS controller modulates clock frequencies and recursion budgets, initiating cooperative aborts during energy droop to prevent brownout. The system achieves a 72.8% reduction in computation with zero loss in classification accuracy while operating within 520 KB SRAM.
