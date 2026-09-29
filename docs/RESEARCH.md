# Internet analysis: porting the RLM to ESP32 with Asynchronous Adaptive Gating

This document is the full internet research synthesis behind the ESP32 port.
Every external fact has a citation; every design decision in the implementation
maps to a findings section. Date of research: August 2026.

---

## 1. Target-hardware landscape

**ESP32 classic vs ESP32-S3 vs the rest.** The M.Tech ECE lab boards are dual-core
Xtensa parts. The classic ESP32 (2 x Xtensa LX6 @ 240 MHz, 520 KB SRAM, FPU + DSP
instructions) and ESP32-S3 (2 x Xtensa LX7 @ 240 MHz, 512 KB SRAM, optional
8 MB PSRAM, **vector/SIMD extensions for ML**) are the realistic targets; RISC-V
parts (C3) lack SIMD and Classic Bluetooth, and STM32H7 wins only at float32 DSP
[1](https://esp32.co.uk/esp32-vs-esp32-s3-which-one-should-you-choose-in-2026/)
[2](https://www.adaptnxt.com/blogs/difference-between-esp32-microcontrollers)
[3](https://www.foresthub.ai/resources/guides/esp32-vs-stm32-for-ai).
The S3 runs TFLM kernels 3-8x faster than the classic on quantised models
[1](https://esp32.co.uk/esp32-vs-esp32-s3-which-one-should-you-choose-in-2026/)
[4](https://zbotic.in/esp32-edge-ai-run-tensorflow-lite-micro-on-microcontroller/).

**Decision (D1).** Write the engine as portable C++ validated on the classic ESP32
DevKit (the most common lab board) and identical source that also builds for
ESP32-S3. We do the timing/parity bench on the classic chip's worst case; S3 owners
get the same firmware with optional SIMD upside. `platformio.ini` ships both envs.

## 2. On-device inference stacks: TFLM / ESP-DL / Edge Impulse / custom

- **TFLite Micro** is the portable runtime with ESP32-S3-optimised kernels written
  with Google, handling conv/dense/pool/activations/quant ops with a static tensor
  arena and no dynamic allocation
  [4](https://zbotic.in/esp32-edge-ai-run-tensorflow-lite-micro-on-microcontroller/)
  [5](https://gizantech.com/blog/edge-ai-tinyml-on-esp32-on-device-machine-learning).
- **ESP-DL** is Espressif's own SIMD-tuned library, strongest for vision kernels on
  S3 [5](https://gizantech.com/blog/edge-ai-tinyml-on-esp32-on-device-machine-learning).
- **Edge Impulse** is a data-to-deployment pipeline, not a research-grade runtime
  for custom architectures
  [5](https://gizantech.com/blog/edge-ai-tinyml-on-esp32-on-device-machine-learning).

None of them express *data-dependent control flow* well: our model loops a
weight-shared block 2..8 times with halting decided on the fly (and, on hardware,
aborted asynchronously by a second core). TFLM graphs are static; graph-cut
surgery would either unroll 8 fixed blocks (8x flash, no halting) or require
custom ops anyway. The shipping TinyML evidence also says quantised inference -
not training - is the MCU regime
[5](https://gizantech.com/blog/edge-ai-tinyml-on-esp32-on-device-machine-learning).

**Decision (D2).** A custom ~350-line C engine with **static buffers, zero heap,
float32 activations, int8 weights dequantised on the fly**. The same engine file
compiles natively on x86 for golden-vector parity (our test harness) and under
Arduino for the chip - so the firmware's numerics are verified, not assumed.

## 3. Adaptive-depth literature (what AAGM adopts and what it discards)

- **ACT (Graves 2016)** adds per-step sigmoidal halting units; models halt when the
  cumulative halting mass reaches `1 - eps`, regularised by a differentiable
  *ponder cost* on the number/mass of steps
  [6](https://arxiv.org/pdf/1603.08983)
  [7](https://www.emergentmind.com/topics/adaptive-computation-time-act).
- **Universal Transformer** applies ACT per-symbol to a weight-shared transformer
  block - the closest public cousin of the repo's RLM
  [8](https://moocaholic.medium.com/adaptive-computation-time-act-in-neural-networks-3-3-99452b2eff18).
- **PonderNet** fixes ACT's instability with a conditional halt distribution and
  exact expected-step gradients, at the cost of stochastic halting
  [9](https://ar5iv.labs.arxiv.org/html/2107.05407).
- **DeeBERT / DACT-BERT / OdeBERT** show early exits (off-ramps / ACT on BERT)
  save ~40-47% inference time at minimal accuracy cost on classification
  [10](https://www.emergentmind.com/papers/2004.12993)
  [11](https://openreview.net/pdf?id=wKfXaxPist)
  [12](https://dl.acm.org/doi/10.1145/3587464).

**Analysis for the RLM.** The repo model mixed states with gate g_k but used a
single-step threshold (gate < 0.5) that, measured in our runs, effectively never
halts (the gate doubles as a small "step size": mean mass accumulates ~0.1-0.2 per
step). On an MCU, *never halting* = paying 8/8 steps always. We adopt ACT's
cumulative mass rule - `halt when sum(1 - g_k) >= 1 - eps` - which turns the
existing soft gate into a hard stopper without changing the mix rule, plus a small
ponder term to make the model commit (Graves' time penalty)
[6](https://arxiv.org/pdf/1603.08983). We reject PonderNet's stochastic halting
(bare-metal determinism is required for bit-level firmware parity and lab
reproducibility) [9](https://ar5iv.labs.arxiv.org/html/2107.05407).
DeeBERT-style early-exit classifiers were rejected for a different reason: a second
head per step duplicates classification weights on an already flash-tight chip;
our **shadow gate** is a 97-parameter *halt forecaster* instead of a per-step
classifier.

**Decision (D3).** ACT-cumulative halting + ponder cost (software AAGM v1), measured:
mean steps 8 -> 2.18 at equal accuracy (artifacts `esp32_software_results.json`).

**Shadow forecaster analysis (why it is honest).** First implemented as
skip-the-gate-MLP speculation, measurements showed the gate MLP is ~5 K MAC vs
~14 M MAC per recursion step (0.03%) - skipping it buys nothing, and skipping
degrades the state trajectory (using g=1 instead of a small gate obliterates the
mix). The mechanism was redesigned to what profiling says is valuable: a
**zero-side-effect scheduling hint** (retire samples early from batches, pre-arm
the output path), calibrated to >= 0.97 precision with full sweep data published
in `calibration.sweep` of the artifact. This "invariant-preserving speculative
scheduler" framing follows the systems lesson of DeeBERT (off-ramps help when they
are cheap and correct) [10](https://www.emergentmind.com/papers/2004.12993).

## 4. Vocabulary without a vocabulary (tokenizer on 520 KB SRAM)

Feature hashing (Weinberger et al.; "the hashing trick") maps tokens via a hash
function into a fixed bucket space with no stored dictionary - constant memory,
streaming-friendly, at the cost of collisions whose impact is "surprisingly small"
for sparse text [13](https://apxml.com/courses/nlp-fundamentals/chapter-2-nlp-feature-engineering/feature-hashing-intro).
spaCy's production embedding layer is exactly this: MultiHashEmbed replaces stored
per-word vectors with hashed rows from a small table, including unknown words, at
comparable downstream accuracy [14](https://arxiv.org/pdf/2212.09255).
Collision-budget analysis for our corpus (~350 distinct tokens): N buckets ->
expected colliding pairs ~ N^2/(2M). With M = 2047 that is ~30 pairs and lexicon
probes confirm damage (sentiment words colliding into the opposite polarity);
with M = 4095 it is ~8 pairs - a 4x reduction for +197 KB of int8 embedding.

**Decision (D4).** FNV-1a 32-bit -> `id = h % 4095 + 1`, PAD 0, embedding
4096 rows (393,216 B int8). Tokenizer implemented **twice, once**, in C and Python
with identical byte semantics (ASCII [a-z0-9'] runs, ASCII lowercasing); the parity
harness asserts equal token ids (caught 0 mismatches). Sketching/heap tricks that
keep a heavy-hitter table [15](https://dawnd9.sites.stanford.edu/news/sketching-classifiers-limited-memory-or-better-feature-hashing-one-simple-trick)
were rejected: reject anything that needs a vocabulary on device.

## 5. Quantisation scheme

LiteRT's 8-bit spec: weights int8 in [-127, 127] with **zero-point 0** (symmetric -
enforced because multiplying the weight zero-point with activations is runtime cost
with no accuracy gain); per-axis scales at the output-channel granularity give
"large improvements to accuracy" without performance implications
[16](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec).
Per-channel int8 typically loses 1-3% accuracy for a 4x size and 2-4x speed win
[4](https://zbotic.in/esp32-edge-ai-run-tensorflow-lite-micro-on-microcontroller/)
[17](https://mbrenndoerfer.com/writing/weight-quantization-basics-scale-zero-point-calibration).

**Decision (D5).** Symmetric per-channel int8 for every weight matrix (incl. the
embedding table, per row); float32 biases/LayerNorms (27 KiB total). Activations
stay float32: Xtensa has a hardware FPU, LayerNorm/softmax in int8 buys little at
this scale, and float accumulation keeps the host-mirror == firmware identity
provable. Measured gap on the test set: **0.00 accuracy points** (mirror 92.33% =
float 92.33%); max logit drift 4.5e-6 end-to-end.

## 6. Dual-core concurrency for the arbiter (hardware AAGM)

Lab-standard pattern: pin tasks with `xTaskCreatePinnedToCore`, communicate via
FreeRTOS primitives; queues for data passing, **direct task notifications as the
zero-RAM-cost 1:1 signalling mechanism** (table in
[18](https://www.wavtron.in/blog/freertos-esp32-multitasking)), and keep heavy
compute off CPU0 where Wi-Fi/BT stacks live - inference on the app core, comms and
system tasks on the pro core
[19](https://zbotic.in/esp32-freertos-tasks-concurrent-iot-operations-explained/)
[20](https://randomnerdtutorials.com/esp32-freertos-queues-inter-task-arduino/)
[21](https://zbotic.in/dual-core-programming-arduino-vs-esp32-concurrency-guide/)
[22](https://zbotic.in/esp32-dual-core-programming-split-tasks-across-both-cores/).

**Decision (D6).** Compute pipeline on core 1; AAGM arbiter task on core 0
(beside Wi-Fi) - evaluates gates via a zero-copy mailbox + task notifications,
samples the battery every 400 ms, parses serial commands asynchronously. Engine
launches the gate then continues state-refinement, so the arbiter's round trip is
hidden ("launch/wait overlap", provider vtable in `rlm_engine.h`).

## 7. Power & DVFS numbers used by the energy gate

Measured draws (dual-core, RF off): 240 MHz ~30-68 mA, 160 MHz ~27-44 mA,
80 MHz ~20-31 mA; a bench log: 66.8 mA @ 240 -> 45.9 mA @ 160 -> 33.2 mA @ 80 ->
19.9 mA @ 40 MHz via `setCpuFrequencyMhz`
[23](https://lastminuteengineers.com/esp32-sleep-modes-power-consumption/)
[24](https://mischianti.org/esp32-practical-power-saving-manage-wifi-and-cpu-1/).
Wi-Fi Tx dwarfs CPU (180-240 mA peaks), so RF stays out of the inference path;
deep sleep is ~10 uA for duty-cycled nodes
[25](https://dronebotworkshop.com/esp32-low-power/).

**Decision (D7).** Profiles PERF 240 MHz/budget 8, BAL 160/6, ECO 80/4 (DVFS +
budget shrink, abort-at-checkpoint semantics), battery telemetry from GPIO34
divider (100k/100k) or simulated mV, `MODE AUTO` fully policy-driven. Energy per
inference scales ~ steps x ms-per-step: adaptive halting to ~2.2 steps is the
dominant saving, ECO DVFS the second multiplier. The energy figure in
`artifacts/esp32_energy_model.png` is annotated model-based; README documents how
to calibrate it with an INA219.

## 8. Novelty claim (survey-checked)

To the best of our survey (TinyML/TFLM/ESP-DL deployment guides
[4](https://zbotic.in/esp32-edge-ai-run-tensorflow-lite-micro-on-microcontroller/)
[5](https://gizantech.com/blog/edge-ai-tinyml-on-esp32-on-device-machine-learning),
ACT/dynamic-halting literature
[6](https://arxiv.org/pdf/1603.08983)
[7](https://www.emergentmind.com/topics/adaptive-computation-time-act)
[8](https://moocaholic.medium.com/adaptive-computation-time-act-in-neural-networks-3-3-99452b2eff18)
[9](https://ar5iv.labs.arxiv.org/html/2107.05407), early-exit transformer systems
[10](https://www.emergentmind.com/papers/2004.12993)
[11](https://openreview.net/pdf?id=wKfXaxPist)
[12](https://dl.acm.org/doi/10.1145/3587464)), **no published work deploys an
ACT-style recursive language model on a microcontroller, and none couples
(i) off-core asynchronous gate evaluation, (ii) a calibrated halt-forecaster
scheduler hint, and (iii) a battery/DVFS-driven recursion budget with
mid-inference cooperative abort as one mechanism** (AAGM). The per-piece building
blocks are properly credited above; the composition and the measured parity
methodology (host-compiled firmware + golden vectors) are the contribution.

## 9. What was tried and rejected (anti-redundancy log)

| idea | why rejected |
|---|---|
| TFLM/ESP-DL graph export | no data-dependent recursion; static graphs; would unroll 8 blocks (8x flash) |
| skip-the-full-gate speculation | saves 0.03% MAC/step, perturbs mixing state (measured) |
| int8 activations | LayerNorm/softmax risk, no measurable speed need (FPU), breaks mirror==firmware identity |
| per-step exit classifiers (DeeBERT-style) | duplicates classifier heads; flash-tight; shadow gate is 97 parameters |
| PonderNet stochastic halting | non-deterministic, blocks bit-parity methodology |
| explicit vocabulary on device | kb of flash + OOV failures; hash embeddings are the production-proven answer |
| sketch/heap heavy-hitter table | needs stored vocab (see D4) |
