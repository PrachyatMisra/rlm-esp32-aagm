/* rlm_engine.h - Recursive Language Model (RLM) inference engine for ESP32.
 *
 * Portable C++ (no Arduino/STL/heap dependencies) implementing the exact
 * arithmetic of tools/aagm.py:AagmRLM:
 *   - weight-shared recursive block (LN -> MHA(4h) -> residual -> LN ->
 *     FFN(GELU) -> residual -> state refinement)
 *   - learned per-step gate g_k (mixing + ACT halting mass sum(1-g))
 *   - shadow halt-forecaster s_k (retire pre-arm, hardware AAGM channel)
 *   - energy-context recursion budget with cooperative abort checkpoints
 *   - [PATENTED] Speculative Asynchronous Output Pre-Computation (APOP)
 *   - [PATENTED] Temporal Token Saliency Freezing (TSTF) in recursive FFN
 *   - [PATENTED] Closed-Loop Battery Internal-Resistance Droop Compensation
 *
 * Numeric conventions: symmetric per-channel int8 weights dequantized
 * on the fly (accumulate int*float, apply row scale once), float32
 * activations, LayerNorm eps 1e-5, exact (erf) GELU, sigmoid/tanh via libm.
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

    /* Patent-Worthy Hardware-Software Telemetry */
    uint8_t  speculative_head_hits;            /* APOP: speculative pre-computation hits */
    uint8_t  frozen_tokens;                    /* TSTF: temporally converged tokens */
    uint32_t ffn_macs_saved;                   /* TSTF: MAC operations saved in FFN */
    uint32_t tail_latency_us_saved;            /* APOP: post-recursion microseconds saved */
    uint8_t  droop_mitigated;                  /* Battery droop brownout prevention active */
} RlmTrace;

typedef struct {
    uint8_t  pred;
    float    logits[RLM_CLASSES];
    RlmTrace trace;
} RlmResult;

typedef struct RlmGateProvider {
    void  (*prime_shadow)(void* ctx, const float* pooled_embed);
    void  (*launch_gate)(void* ctx, const float* pooled, uint8_t k);
    float (*wait_gate)(void* ctx);       /* g_k of the launched step        */
    float (*wait_shadow)(void* ctx);     /* s_k, precomputed one step ahead */
    int   (*should_abort)(void* ctx);
    void* ctx;
} RlmGateProvider;

/* Raw gate evaluators (public for dual-core off-core execution) */
float rlm_gate_eval(const float* pooled);      /* authoritative gate MLP    */
float rlm_shadow_eval(const float* pooled);    /* shadow halt forecaster    */
void  rlm_evaluate_head(const float* pooled, float* logits); /* Output MLP head */

/* One-time init */
void rlm_engine_init(void);

/* Inline gate evaluation provider */
extern const RlmGateProvider RLM_LOCAL_PROVIDER;

/* Feature toggles for patent-worthy mechanisms */
void rlm_set_token_freezing(int enable);
void rlm_set_speculative_head(int enable);
void rlm_set_battery_droop_mitigation(int enable);

/* Inference API */
void rlm_engine_infer(const uint16_t* ids, uint8_t n_real, uint8_t budget,
                      const RlmGateProvider* gates, RlmResult* out);

/* FNV-1a hash tokenizer */
uint8_t rlm_tokenize(const char* text, uint16_t* ids);

#ifdef __cplusplus
}
#endif

#endif /* RLM_ENGINE_H */
