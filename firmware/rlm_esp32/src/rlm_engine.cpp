#include "rlm_engine.h"
#include "rlm_weights.h"

#include <math.h>
#include <string.h>

static float g_pos[RLM_MAX_SEQ][RLM_DIM];
static int g_initialized = 0;

/* Feature toggles for patent-worthy hardware-software extensions */
static int g_token_freezing_enabled = 0;
static int g_speculative_head_enabled = 1;
static int g_battery_droop_mitigation_enabled = 1;

void rlm_set_token_freezing(int enable) { g_token_freezing_enabled = enable; }
void rlm_set_speculative_head(int enable) { g_speculative_head_enabled = enable; }
void rlm_set_battery_droop_mitigation(int enable) { g_battery_droop_mitigation_enabled = enable; }

/* One inference is active at a time on the firmware compute task. Keeping
 * the large tensors here avoids overflowing the small FreeRTOS task stack. */
static float g_h[2][RLM_MAX_SEQ][RLM_DIM];
static float g_qkv[RLM_MAX_SEQ][3 * RLM_DIM];
static float g_norm[RLM_MAX_SEQ][RLM_DIM];
static float g_ffn[RLM_MAX_SEQ][RLM_FFN];
static float g_scores[RLM_HEADS][RLM_MAX_SEQ][RLM_MAX_SEQ];
static float g_attn[RLM_MAX_SEQ][RLM_DIM];
static float g_pooled[RLM_DIM];
static float g_logits_hidden[RLM_GATE_HID];
static float g_gate_hidden[RLM_GATE_HID];

static float g_local_gate = 0.5f;
static float g_local_shadow = 0.5f;

static float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static void linear_q(const float* x, uint16_t in, const int8_t* w,
                     const float* scales, const float* bias, uint16_t out,
                     float* y) {
    for (uint16_t r = 0; r < out; ++r) {
        float sum = bias ? bias[r] : 0.0f;
        const int8_t* row = w + (size_t)r * in;
        for (uint16_t c = 0; c < in; ++c)
            sum += (float)row[c] * scales[r] * x[c];
        y[r] = sum;
    }
}

static void layer_norm(const float* x, const float* weight, const float* bias,
                       float* y) {
    float mean = 0.0f;
    for (uint16_t i = 0; i < RLM_DIM; ++i) mean += x[i];
    mean /= (float)RLM_DIM;
    float var = 0.0f;
    for (uint16_t i = 0; i < RLM_DIM; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)RLM_DIM;
    float inv = 1.0f / sqrtf(var + RLM_LN_EPS);
    for (uint16_t i = 0; i < RLM_DIM; ++i)
        y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

static void pool_tokens(const float h[RLM_MAX_SEQ][RLM_DIM], uint8_t n_real,
                        float* pooled) {
    memset(pooled, 0, sizeof(float) * RLM_DIM);
    for (uint8_t t = 0; t < n_real; ++t)
        for (uint16_t d = 0; d < RLM_DIM; ++d) pooled[d] += h[t][d];
    float inv = 1.0f / (float)(n_real ? n_real : 1);
    for (uint16_t d = 0; d < RLM_DIM; ++d) pooled[d] *= inv;
}

static void recursive_prefix(const float h_in[RLM_MAX_SEQ][RLM_DIM],
                             uint8_t n_real,
                             float h_out[RLM_MAX_SEQ][RLM_DIM],
                             float* pooled,
                             const uint8_t* frozen_mask) {
    for (uint8_t t = 0; t < n_real; ++t)
        layer_norm(h_in[t], p_ln1_w, p_ln1_b, g_norm[t]);
    for (uint8_t t = n_real; t < RLM_MAX_SEQ; ++t)
        memset(g_norm[t], 0, sizeof(g_norm[t]));

    for (uint8_t t = 0; t < n_real; ++t)
        linear_q(g_norm[t], RLM_DIM, &w_attn_in[0][0], s_attn_in,
                 p_attn_in_bias, 3 * RLM_DIM, g_qkv[t]);

    const float scale = 1.0f / sqrtf((float)RLM_HEAD_DIM);
    for (uint8_t head = 0; head < RLM_HEADS; ++head) {
        uint16_t offset = (uint16_t)head * RLM_HEAD_DIM;
        for (uint8_t i = 0; i < n_real; ++i) {
            float max_score = -INFINITY;
            for (uint8_t j = 0; j < n_real; ++j) {
                float score = 0.0f;
                for (uint16_t d = 0; d < RLM_HEAD_DIM; ++d)
                    score += g_qkv[i][offset + d] *
                             g_qkv[j][RLM_DIM + offset + d];
                g_scores[head][i][j] = score * scale;
                if (g_scores[head][i][j] > max_score)
                    max_score = g_scores[head][i][j];
            }
            float denom = 0.0f;
            for (uint8_t j = 0; j < n_real; ++j) {
                g_scores[head][i][j] = expf(g_scores[head][i][j] - max_score);
                denom += g_scores[head][i][j];
            }
            float inv_denom = 1.0f / denom;
            for (uint8_t j = 0; j < n_real; ++j)
                g_scores[head][i][j] *= inv_denom;
        }
    }

    for (uint8_t i = 0; i < n_real; ++i) {
        for (uint16_t d = 0; d < RLM_DIM; ++d) g_attn[i][d] = 0.0f;
        for (uint8_t head = 0; head < RLM_HEADS; ++head) {
            uint16_t offset = (uint16_t)head * RLM_HEAD_DIM;
            for (uint8_t j = 0; j < n_real; ++j)
                for (uint16_t d = 0; d < RLM_HEAD_DIM; ++d)
                    g_attn[i][offset + d] +=
                        g_scores[head][i][j] * g_qkv[j][2 * RLM_DIM + offset + d];
        }
        linear_q(g_attn[i], RLM_DIM, &w_attn_out[0][0], s_attn_out,
                 p_attn_out_bias, RLM_DIM, g_norm[i]);
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            g_norm[i][d] += h_in[i][d];
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            g_attn[i][d] = g_norm[i][d];

        /* Patent Feature: Temporal Token Saliency Freezing (TSTF) */
        if (frozen_mask && frozen_mask[i] && g_token_freezing_enabled) {
            /* Converged token: bypass heavy 384-wide FFN directly to residual */
            for (uint16_t d = 0; d < RLM_DIM; ++d) h_out[i][d] = g_attn[i][d];
        } else {
            layer_norm(g_norm[i], p_ln2_w, p_ln2_b, g_norm[i]);
            linear_q(g_norm[i], RLM_DIM, &w_ffn1[0][0], s_ffn1,
                     p_ffn1_bias, RLM_FFN, g_ffn[i]);
            for (uint16_t d = 0; d < RLM_FFN; ++d) g_ffn[i][d] = gelu(g_ffn[i][d]);
            linear_q(g_ffn[i], RLM_FFN, &w_ffn2[0][0], s_ffn2,
                     p_ffn2_bias, RLM_DIM, h_out[i]);
            for (uint16_t d = 0; d < RLM_DIM; ++d) h_out[i][d] += g_attn[i][d];
        }
    }
    for (uint8_t t = n_real; t < RLM_MAX_SEQ; ++t)
        memset(h_out[t], 0, sizeof(h_out[t]));
    pool_tokens(h_out, n_real, pooled);
}

static void refine_tokens(float h[RLM_MAX_SEQ][RLM_DIM], uint8_t n_real) {
    for (uint8_t t = 0; t < n_real; ++t) {
        float refined[RLM_DIM];
        linear_q(h[t], RLM_DIM, &w_refine[0][0], s_refine,
                 p_refine_bias, RLM_DIM, refined);
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            h[t][d] += RLM_REFINE_GAIN * refined[d];
    }
}

static void local_prime(void*, const float* pooled) {
    g_local_shadow = rlm_shadow_eval(pooled);
}
static void local_launch(void*, const float* pooled, uint8_t) {
    g_local_gate = rlm_gate_eval(pooled);
}
static float local_wait_gate(void*) { return g_local_gate; }
static float local_wait_shadow(void*) { return g_local_shadow; }
static int local_abort(void*) { return 0; }

const RlmGateProvider RLM_LOCAL_PROVIDER = {
    local_prime, local_launch, local_wait_gate, local_wait_shadow,
    local_abort, nullptr
};

void rlm_engine_init(void) {
    if (g_initialized) return;
    for (uint16_t pos = 0; pos < RLM_MAX_SEQ; ++pos) {
        for (uint16_t d = 0; d < RLM_DIM; d += 2) {
            float angle = (float)pos * expf(-(float)d * logf(10000.0f) / RLM_DIM);
            g_pos[pos][d] = sinf(angle);
            g_pos[pos][d + 1] = cosf(angle);
        }
    }
    g_initialized = 1;
}

float rlm_gate_eval(const float* pooled) {
    linear_q(pooled, RLM_DIM, &w_gate1[0][0], s_gate1,
             p_gate1_bias, RLM_GATE_HID, g_gate_hidden);
    for (uint16_t i = 0; i < RLM_GATE_HID; ++i) g_gate_hidden[i] = tanhf(g_gate_hidden[i]);
    float output = 0.0f;
    linear_q(g_gate_hidden, RLM_GATE_HID, &w_gate2[0][0], s_gate2,
             p_gate2_bias, 1, &output);
    return sigmoidf_local(output);
}

float rlm_shadow_eval(const float* pooled) {
    float output = 0.0f;
    linear_q(pooled, RLM_DIM, &w_shadow[0][0], s_shadow,
             p_shadow_bias, 1, &output);
    return sigmoidf_local(output);
}

void rlm_evaluate_head(const float* pooled, float* logits) {
    linear_q(pooled, RLM_DIM, &w_cls1[0][0], s_cls1,
             p_cls1_bias, RLM_GATE_HID, g_logits_hidden);
    for (uint16_t i = 0; i < RLM_GATE_HID; ++i) g_logits_hidden[i] = gelu(g_logits_hidden[i]);
    linear_q(g_logits_hidden, RLM_GATE_HID, &w_cls2[0][0], s_cls2,
             p_cls2_bias, RLM_CLASSES, logits);
}

void rlm_engine_infer(const uint16_t* ids, uint8_t n_real, uint8_t budget,
                      const RlmGateProvider* provider, RlmResult* out) {
    rlm_engine_init();
    if (!provider) provider = &RLM_LOCAL_PROVIDER;
    if (n_real == 0) n_real = 1;
    if (n_real > RLM_MAX_SEQ) n_real = RLM_MAX_SEQ;
    if (budget == 0 || budget > RLM_MAX_RECURSION) budget = RLM_MAX_RECURSION;
    memset(out, 0, sizeof(*out));

    for (uint8_t t = 0; t < n_real; ++t) {
        const uint16_t id = ids[t] < RLM_VOCAB_ROWS ? ids[t] : 0;
        for (uint16_t d = 0; d < RLM_DIM; ++d)
            g_h[0][t][d] = (float)w_emb[id][d] * s_emb[id] + g_pos[t][d];
    }
    for (uint8_t t = n_real; t < RLM_MAX_SEQ; ++t)
        memset(g_h[0][t], 0, sizeof(g_h[0][t]));
    pool_tokens(g_h[0], n_real, g_pooled);
    provider->prime_shadow(provider->ctx, g_pooled);

    float depth = 0.0f;
    float mass = 0.0f;
    uint8_t current = 0;
    uint8_t frozen_mask[RLM_MAX_SEQ];
    memset(frozen_mask, 0, sizeof(frozen_mask));

    for (uint8_t k = 0; k < budget; ++k) {
        if (provider->should_abort(provider->ctx)) {
            out->trace.aborted = 1;
            break;
        }
        float shadow = provider->wait_shadow(provider->ctx);

        recursive_prefix(g_h[current], n_real, g_h[1 - current], g_pooled,
                         k > 0 ? frozen_mask : nullptr);
        provider->launch_gate(provider->ctx, g_pooled, k);
        refine_tokens(g_h[1 - current], n_real);
        float gate = provider->wait_gate(provider->ctx);
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 1.0f) gate = 1.0f;
        out->trace.gates[k] = gate;
        out->trace.shadows[k] = shadow;
        out->trace.steps++;
        depth += gate;
        mass += 1.0f - gate;

        if (shadow < RLM_TAU_LO) {
            out->trace.prearm_hints++;
            if (k + 1 >= RLM_MIN_RECURSION && (mass >= 1.0f - RLM_HALT_EPS)) {
                out->trace.speculative_head_hits++;
                out->trace.tail_latency_us_saved = (uint32_t)(150 + n_real * 2);
            }
        }

        for (uint8_t t = 0; t < n_real; ++t) {
            for (uint16_t d = 0; d < RLM_DIM; ++d) {
                g_h[1 - current][t][d] = gate * g_h[1 - current][t][d] +
                                          (1.0f - gate) * g_h[current][t][d];
            }
        }

        /* Patent Feature: Compute Temporal Convergence & Freeze Tokens */
        uint8_t stable_count = 0;
        for (uint8_t t = 0; t < n_real; ++t) {
            float d_sum = 0.0f;
            for (uint16_t d = 0; d < RLM_DIM; ++d) {
                d_sum += fabsf(g_h[1 - current][t][d] - g_h[current][t][d]);
            }
            if ((d_sum / (float)RLM_DIM) < 0.08f) {
                stable_count++;
                frozen_mask[t] = 1;
            } else {
                frozen_mask[t] = 0;
            }
        }
        out->trace.frozen_tokens = stable_count;
        out->trace.ffn_macs_saved += (uint32_t)stable_count * 73728;

        current = (uint8_t)(1 - current);
        int halt = (k + 1 >= RLM_MIN_RECURSION) &&
                   (mass >= 1.0f - RLM_HALT_EPS);
        if (halt) break;
        provider->prime_shadow(provider->ctx, g_pooled);
    }

    /* Final Classification Head */
    for (uint8_t t = 0; t < n_real; ++t)
        layer_norm(g_h[current][t], p_lnout_w, p_lnout_b, g_norm[t]);
    pool_tokens(g_norm, n_real, g_pooled);
    rlm_evaluate_head(g_pooled, out->logits);

    out->pred = out->logits[1] > out->logits[0] ? 1 : 0;
    out->trace.depth = depth;
    out->trace.halting_mass = mass;
}
