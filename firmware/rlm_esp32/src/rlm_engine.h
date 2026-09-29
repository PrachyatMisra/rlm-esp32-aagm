/* rlm_engine.h - Recursive Language Model (RLM) inference engine for ESP32.
 *
 * Portable C++ (no Arduino/STL/heap dependencies) implementing the exact
 * arithmetic of tools/aagm.py:AagmRLM:
 *   - weight-shared recursive block (LN -> MHA(4h) -> residual -> LN ->
 *     FFN(GELU) -> residual -> state refinement)
 *   - learned per-step gate g_k (mixing + ACT halting mass sum(1-g))
 *   - shadow halt-forecaster s_k (retire pre-arm, hardware AAGM channel)
 *   - energy-context recursion budget with cooperative abort checkpoints
 *
 * Numeric conventions: symmetric per-channel int8 weights dequantized
 * on the fly (accumulate int*float, apply row scale once), float32
 * activations, LayerNorm eps 1e-5, exact (erf) GELU, sigmoid/tanh via libm.
 *
 * The engine never evaluates gating policy internally at step end; it calls
 * an RlmGateProvider. Two providers exist:
 *   - rlm_local_provider (rlm_engine.cpp): inline eval (host harness/tests)
 *   - the FreeRTOS mailbox arbiter in rlm_esp32.ino (off-core async eval)
 */
#ifndef RLM_ENGINE_H
#define RLM_ENGINE_H

#include <stdint.h>
#include "rlm_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t  steps;                            /* recursion steps executed */
    uint8_t  prearm_hints;                     /* shadow retire hints fired */
    uint8_t  aborted;                          /* stopped by async abort */
    float    depth;                            /* soft depth: sum of gates  */
    float    halting_mass;                     /* ACT mass at termination   */
    float    gates[RLM_MAX_RECURSION];         /* g_k actually used         */
    float    shadows[RLM_MAX_RECURSION];       /* s_k forecasts             */
} RlmTrace;

typedef struct {
    uint8_t  pred;
    float    logits[RLM_CLASSES];
    RlmTrace trace;
} RlmResult;

typedef struct RlmGateProvider {
    /* Called once after embedding with the initial pooled state so the
     * provider can precompute s_0 (arbiter: async round trip to core 0). */
    void  (*prime_shadow)(void* ctx, const float* pooled_embed);
    /* Called as soon as the pooled pre-refine state of step k exists. Must
     * return promptly: the engine runs the state-refinement stage while the
     * provider resolves the gate (launch/wait overlap). */
    void  (*launch_gate)(void* ctx, const float* pooled, uint8_t k);
    float (*wait_gate)(void* ctx);       /* g_k of the launched step        */
    float (*wait_shadow)(void* ctx);     /* s_k, precomputed one step ahead */
    /* Cooperative abort checkpoint polled before each recursion step and
     * inside the block (attention / FFN boundaries). Nonzero -> stop now,
     * keep current h, mark trace.aborted. The energy arbiter sets this when
     * the recursion budget shrinks mid-inference. */
    int   (*should_abort)(void* ctx);
    void* ctx;
} RlmGateProvider;

/* Raw gate evaluators (public so the firmware arbiter can run them on the
 * other core without duplicating gate arithmetic here). */
float rlm_gate_eval(const float* pooled);      /* authoritative gate MLP    */
float rlm_shadow_eval(const float* pooled);    /* shadow halt forecaster    */

/* One-time init: builds the sinusoidal positional table (matches the
 * float32 construction in tools/aagm.py). Call before first inference. */
void rlm_engine_init(void);

/* Inline gate evaluation on the calling core (host harness / benchmarks). */
extern const RlmGateProvider RLM_LOCAL_PROVIDER;

/* ids: token ids (1..RLM_VOCAB_ROWS-1), n_real: number of real tokens
 * (1..RLM_MAX_SEQ), budget: energy-context cap (1..RLM_MAX_RECURSION). */
void rlm_engine_infer(const uint16_t* ids, uint8_t n_real, uint8_t budget,
                      const RlmGateProvider* gates, RlmResult* out);

/* FNV-1a hash tokenizer (identical to tools/aagm.py:tokenize). Fills ids
 * (size RLM_MAX_SEQ, zero-padded), returns number of real tokens. */
uint8_t rlm_tokenize(const char* text, uint16_t* ids);

#ifdef __cplusplus
}
#endif

#endif /* RLM_ENGINE_H */
