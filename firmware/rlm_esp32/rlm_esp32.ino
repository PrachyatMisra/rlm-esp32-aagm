/*
 * rlm_esp32.ino - Recursive Language Model with Asynchronous Adaptive
 * Gating Mechanism (AAGM) on ESP32 (Arduino core).
 *
 * Hardware-level AAGM (dual-core FreeRTOS):
 *   Core 1 (APP_CPU) - compute task: tokenizer -> recursive block steps ->
 *                      head. Pure math, never handles I/O. Runs INFER/CHAT/BENCH
 *                      jobs taken from a request mailbox.
 *   Core 0 (PRO_CPU) - AAGM arbiter task: evaluates ALL gates (authoritative
 *                      gate + shadow forecaster) offloaded via a zero-copy
 *                      mailbox; samples battery with internal resistance droop
 *                      compensation; applies DVFS energy-context policy
 *                      (PERF 240MHz/B8, BAL 160MHz/B6, ECO 80MHz/B4); and
 *                      parses async serial commands. Mid-inference budget shrink
 *                      or battery droop triggers a cooperative abort checkpoint.
 *
 * Serial protocol (115200 baud, newline-terminated ASCII):
 *   CHAT <text>                 -> conversational RLM response with reasoning
 *   INFER <text>                -> raw JSON telemetry line
 *   MODE PERF|BAL|ECO|AUTO      (AUTO derives profile from compensated battery)
 *   BATT <millivolts>|AUTO      (AUTO: GPIO34 100k/100k divider x2)
 *   BENCH <n>                   -> n self-timing runs of a fixed review
 *   STAT                        -> heap/freq/board state
 *   PING                        -> {"pong":1,...}
 */
#include <Arduino.h>
#include "src/rlm_engine.h"

/* ---------------- energy context / profiles --------------------------- */
enum Profile : uint8_t { PROF_PERF = 0, PROF_BAL = 1, PROF_ECO = 2 };
static const uint8_t  PROF_BUDGET[3] = { 8, 6, 4 };
static const uint32_t PROF_MHZ[3]    = { 240, 160, 80 };
static const char*    PROF_NAME[3]   = { "PERF", "BAL", "ECO" };

static volatile uint8_t  g_profile   = PROF_PERF;
static volatile bool     g_auto_mode = false;
static volatile uint8_t  g_budget    = RLM_MAX_RECURSION;
static volatile uint32_t g_abort     = 0;

static volatile bool     g_batt_auto = false;
static float             g_batt_mv   = 4000.0f; /* Open-circuit voltage (OCV) */
static const char*       g_batt_src  = "sim";
static const int         BATT_ADC_PIN = 34;     /* 100k/100k divider */

/* Battery internal resistance droop compensation (R_int = 200 mOhm) */
static const float BATT_R_INT = 0.200f;

static SemaphoreHandle_t g_serial_mtx;

/* ---------------- async gate mailbox (core 1 -> core 0) ---------------- */
typedef struct {
    const float* pooled;      /* zero-copy into the engine's buffers        */
    uint8_t      kind;        /* 0 = gate(+shadow), 1 = shadow-only prime   */
    float        gate;        /* out: g_k                                   */
    float        shadow;      /* out: s_{k+1}                               */
} GateJob;
static GateJob      g_job;
static volatile uint32_t g_job_seq = 0, g_job_done = 0;
static TaskHandle_t g_compute_task = nullptr;
static TaskHandle_t g_arbiter_task = nullptr;

/* gate-provider callbacks, executed on the compute core */
static void fw_post(const float* pooled, uint8_t kind) {
    g_job.pooled = pooled;
    g_job.kind   = kind;
    g_job_seq++;                                /* publish request */
    xTaskNotifyGive(g_arbiter_task);
}
static void fw_await() {                        /* spin-free wait for ack */
    while (g_job_done != g_job_seq)
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
}
static void fw_prime(void*, const float* pooled_embed) {
    fw_post(pooled_embed, 1);
    fw_await();
}
static void fw_launch(void*, const float* pooled, uint8_t) {
    fw_post(pooled, 0);                          /* engine refines meanwhile */
}
static float fw_wait_gate(void*) {
    fw_await();
    return g_job.gate;
}
static float fw_wait_shadow(void*) { return g_job.shadow; }
static int   fw_should_abort(void*) { return g_abort != 0; }

static RlmGateProvider FW_PROVIDER = {
    fw_prime, fw_launch, fw_wait_gate, fw_wait_shadow, fw_should_abort, nullptr
};

/* ---------------- battery policy with droop compensation --------------- */
static float compute_loaded_mv(float ocv_mv, uint32_t cpu_mhz) {
    float i_ma = (cpu_mhz >= 240) ? 50.0f : ((cpu_mhz >= 160) ? 36.0f : 27.0f);
    float droop_mv = (i_ma / 1000.0f) * BATT_R_INT * 1000.0f;
    return ocv_mv - droop_mv;
}

static float read_batt_mv() {
    if (!g_batt_auto) return g_batt_mv;
    uint32_t mv = analogReadMilliVolts(BATT_ADC_PIN);
    if (mv < 200) { g_batt_src = "sim(float-adc)"; return 4000.0f; }
    g_batt_src = "adc34";
    return mv * 2.0f;                           /* 100k/100k divider */
}

static void apply_profile(uint8_t prof) {
    uint8_t new_budget = PROF_BUDGET[prof];
    if (new_budget < g_budget) g_abort = 1;     /* shrink -> async abort   */
    g_profile = prof;
    g_budget  = new_budget;
    setCpuFrequencyMhz(PROF_MHZ[prof]);
}

static uint8_t profile_for_batt(float loaded_mv) {
    float pct = constrain((loaded_mv - 3300.0f) / 900.0f * 100.0f, 0.0f, 100.0f);
    if (pct < 25.0f) return PROF_ECO;
    if (pct < 50.0f) return PROF_BAL;
    return PROF_PERF;
}

static void energy_tick() {                     /* runs on the arbiter core */
    float ocv = read_batt_mv();
    g_batt_mv = ocv;
    float loaded = compute_loaded_mv(ocv, PROF_MHZ[g_profile]);

    /* Brownout protection: if loaded voltage approaches cutoff within 50mV */
    if (loaded < 3350.0f) {
        apply_profile(PROF_ECO);
        g_abort = 1; // trigger early exit to prevent brownout
    } else if (g_auto_mode) {
        apply_profile(profile_for_batt(loaded));
    }
}

/* ---------------- job mailbox (arbiter -> compute core) ---------------- */
static char          g_req_text[512];
static volatile int  g_req_infer = 0;           /* 1 pending inference       */
static volatile int  g_req_chat  = 0;           /* 1 pending chat prompt     */
static volatile int  g_req_bench = 0;           /* >0 pending bench runs     */
static volatile bool g_req_stat  = false;

/* ---------------- JSON / inference (compute core) ---------------------- */
static void json_lock()   { xSemaphoreTake(g_serial_mtx, portMAX_DELAY); }
static void json_unlock() { xSemaphoreGive(g_serial_mtx); }

static void do_infer(const char* text) {
    uint16_t ids[RLM_MAX_SEQ];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) { json_lock(); Serial.println("{\"err\":\"empty\"}");
                       json_unlock(); return; }
    g_abort = 0;
    uint32_t t0 = micros();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &r);
    uint32_t us = micros() - t0;

    float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
    float droop = g_batt_mv - loaded;

    json_lock();
    Serial.printf("{\"pred\":%u,\"logits\":[%.4f,%.4f],\"steps\":%u,"
                  "\"depth\":%.4f,\"mass\":%.4f,\"hints\":%u,\"aborted\":%u,"
                  "\"spec_hits\":%u,\"frozen\":%u,\"gates\":[",
                  r.pred, r.logits[0], r.logits[1],
                  r.trace.steps, r.trace.depth, r.trace.halting_mass,
                  r.trace.prearm_hints, r.trace.aborted,
                  r.trace.speculative_head_hits, r.trace.frozen_tokens);
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        Serial.printf(i ? ",%.4f" : "%.4f", r.trace.gates[i]);
    Serial.print("],\"shadows\":[");
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        Serial.printf(i ? ",%.4f" : "%.4f", r.trace.shadows[i]);
    Serial.printf("],\"us\":%lu,\"budget\":%u,\"profile\":\"%s\",\"batt_mv\":%.0f,"
                  "\"loaded_mv\":%.0f,\"droop_mv\":%.1f,\"heap\":%u}\n",
                  (unsigned long)us, g_budget, PROF_NAME[g_profile],
                  (double)g_batt_mv, (double)loaded, (double)droop, ESP.getFreeHeap());
    json_unlock();
}

static void do_chat(const char* text) {
    uint16_t ids[RLM_MAX_SEQ];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        json_lock();
        Serial.println("{\"reply\":\"Please provide text for me to analyze.\"}");
        json_unlock();
        return;
    }
    g_abort = 0;
    uint32_t t0 = micros();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &r);
    uint32_t us = micros() - t0;

    float max_l = max(r.logits[0], r.logits[1]);
    float e0 = expf(r.logits[0] - max_l);
    float e1 = expf(r.logits[1] - max_l);
    float conf = (r.pred == 1 ? e1 : e0) / (e0 + e1) * 100.0f;
    const char* verdict = r.pred == 1 ? "POSITIVE" : "NEGATIVE";

    json_lock();
    Serial.printf("{\"chat_reply\":\"I evaluated your input: '%s'. "
                  "Verdict: %s (Confidence: %.1f%%). "
                  "Reasoned in %u recursive steps (Mass: %.3f/0.900), "
                  "saving %u steps (%.0f%% compute reduction). Latency: %lu us.\","
                  "\"pred\":%u,\"verdict\":\"%s\",\"confidence\":%.1f,"
                  "\"steps\":%u,\"saved_steps\":%u,\"us\":%lu,\"profile\":\"%s\"}\n",
                  text, verdict, conf, r.trace.steps, r.trace.halting_mass,
                  (8 - r.trace.steps), (float)(8 - r.trace.steps) / 8.0f * 100.0f,
                  (unsigned long)us, r.pred, verdict, conf, r.trace.steps,
                  (8 - r.trace.steps), (unsigned long)us, PROF_NAME[g_profile]);
    json_unlock();
}

static void do_bench(int n) {
    const char* text = "the first act is clumsy and uneven meanwhile the cast "
                       "shines, yet the final scene feels magnificent";
    uint16_t ids[RLM_MAX_SEQ];
    uint8_t n_real = rlm_tokenize(text, ids);
    uint32_t best = UINT32_MAX, worst = 0, steps_sum = 0;
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        g_abort = 0;
        uint32_t t0 = micros();
        RlmResult r;
        rlm_engine_infer(ids, n_real, g_budget, &FW_PROVIDER, &r);
        uint32_t us = micros() - t0;
        best = min(best, us); worst = max(worst, us);
        sum += us; steps_sum += r.trace.steps;
    }
    json_lock();
    Serial.printf("{\"bench_n\":%d,\"us_avg\":%.1f,\"us_min\":%lu,\"us_max\":%lu,"
                  "\"mean_steps\":%.3f,\"budget\":%u,\"cpu_mhz\":%lu,"
                  "\"profile\":\"%s\",\"heap\":%u}\n",
                  n, sum / n, (unsigned long)best, (unsigned long)worst,
                  (double)steps_sum / n, g_budget,
                  (unsigned long)getCpuFrequencyMhz(), PROF_NAME[g_profile],
                  ESP.getFreeHeap());
    json_unlock();
}

static void do_stat() {
    float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
    json_lock();
    Serial.printf("{\"board\":\"%s\",\"cpu_mhz\":%lu,\"budget\":%u,"
                  "\"profile\":\"%s\",\"auto_mode\":%s,\"batt_mv\":%.0f,"
                  "\"loaded_mv\":%.0f,\"batt_src\":\"%s\",\"heap\":%u,"
                  "\"min_heap\":%u,\"sketch_kb\":%u,\"flash_mb\":%u}\n",
                  ARDUINO_VARIANT, (unsigned long)getCpuFrequencyMhz(), g_budget,
                  PROF_NAME[g_profile], g_auto_mode ? "true" : "false",
                  (double)g_batt_mv, (double)loaded, g_batt_src, ESP.getFreeHeap(),
                  ESP.getMinFreeHeap(), ESP.getSketchSize() / 1024,
                  ESP.getFlashChipSize() / (1024 * 1024));
    json_unlock();
}

/* ---------------- command handling (arbiter core) ---------------------- */
static void handle_line(char* line) {
    json_lock();
    if (!strncmp(line, "CHAT ", 5)) {
        if (g_req_infer || g_req_chat) { Serial.println("{\"err\":\"busy\"}"); }
        else {
            strncpy(g_req_text, line + 5, sizeof(g_req_text) - 1);
            g_req_text[sizeof(g_req_text) - 1] = 0;
            g_req_chat = 1;
        }
    }
    else if (!strncmp(line, "INFER ", 6)) {
        if (g_req_infer || g_req_chat) { Serial.println("{\"err\":\"busy\"}"); }
        else {
            strncpy(g_req_text, line + 6, sizeof(g_req_text) - 1);
            g_req_text[sizeof(g_req_text) - 1] = 0;
            g_req_infer = 1;
        }
    }
    else if (!strncmp(line, "BENCH ", 6)) {
        if (g_req_infer || g_req_bench || g_req_chat) Serial.println("{\"err\":\"busy\"}");
        else g_req_bench = max(1, atoi(line + 6));
    }
    else if (!strcmp(line, "STAT"))             g_req_stat = true;
    else if (!strcmp(line, "PING"))
        Serial.printf("{\"pong\":1,\"uptime_ms\":%lu}\n", millis());
    else if (!strncmp(line, "MODE ", 5)) {
        const char* m = line + 5;
        if      (!strcmp(m, "PERF")) { g_auto_mode = false; apply_profile(PROF_PERF); }
        else if (!strcmp(m, "BAL"))  { g_auto_mode = false; apply_profile(PROF_BAL); }
        else if (!strcmp(m, "ECO"))  { g_auto_mode = false; apply_profile(PROF_ECO); }
        else if (!strcmp(m, "AUTO")) { g_auto_mode = true; energy_tick(); }
        Serial.printf("{\"mode\":\"%s\",\"budget\":%u,\"cpu_mhz\":%lu}\n",
                      PROF_NAME[g_profile], g_budget,
                      (unsigned long)getCpuFrequencyMhz());
    }
    else if (!strncmp(line, "BATT ", 5)) {
        const char* b = line + 5;
        if (!strcmp(b, "AUTO")) { g_batt_auto = true; }
        else { g_batt_auto = false; g_batt_mv = (float)atof(b); g_batt_src = "sim"; }
        energy_tick();
        Serial.printf("{\"batt_mv\":%.0f,\"batt_src\":\"%s\",\"profile\":\"%s\","
                      "\"budget\":%u}\n", (double)g_batt_mv, g_batt_src,
                      PROF_NAME[g_profile], g_budget);
    }
    else Serial.println("{\"err\":\"unknown cmd\"}");
    json_unlock();
}

/* ---------------- arbiter task (core 0) -------------------------------- */
static void arbiter_loop(void*) {
    char line[512];
    size_t len = 0;
    uint32_t next_energy = millis() + 500;
    for (;;) {
        uint32_t now = millis();
        uint32_t wait_ms = next_energy > now ? next_energy - now : 1;
        if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(wait_ms))) {
            uint32_t seq = g_job_seq;
            float g = 0.f, s = 0.f;
            const float* pooled = g_job.pooled;
            uint8_t kind = g_job.kind;
            if (kind == 0) g = rlm_gate_eval(pooled);
            s = rlm_shadow_eval(pooled);
            g_job.gate = g;
            g_job.shadow = s;
            g_job_done = seq;                        /* publish results */
            xTaskNotifyGive(g_compute_task);         /* wake engine core */
        }
        if ((int32_t)(now - next_energy) >= 0) {     /* battery heartbeat */
            energy_tick();
            next_energy = now + 400;
        }
        while (Serial.available()) {                 /* async host I/O    */
            char c = (char)Serial.read();
            if (c == '\r') continue;
            if (c == '\n' || len >= sizeof(line) - 1) {
                line[len] = 0;
                if (len) handle_line(line);
                len = 0;
            } else {
                line[len++] = c;
            }
        }
    }
}

/* ---------------- boot / dispatcher ------------------------------------ */
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    g_serial_mtx = xSemaphoreCreateMutex();

    rlm_engine_init();
    g_compute_task = xTaskGetCurrentTaskHandle();  /* core 1 (APP_CPU)    */

    xTaskCreatePinnedToCore(arbiter_loop, "aagm_arbiter", 6144, nullptr,
                            3, &g_arbiter_task, 0 /* PRO_CPU, beside Wi-Fi */);

    Serial.printf(">RLM-AAGM fw=2 board=%s vocab=%u dim=%u max_rec=%u "
                  "tau_lo=%.3f budget=%u chat=ready\n", ARDUINO_VARIANT, RLM_VOCAB_ROWS,
                  RLM_DIM, RLM_MAX_RECURSION, (double)RLM_TAU_LO, g_budget);
}

void loop() {                                      /* core 1 job dispatcher */
    if (g_req_infer) { do_infer(g_req_text); g_req_infer = 0; }
    if (g_req_chat)  { do_chat(g_req_text);  g_req_chat  = 0; }
    if (g_req_bench > 0) { int n = g_req_bench; g_req_bench = 0; do_bench(n); }
    if (g_req_stat)  { g_req_stat = false;   do_stat(); }
    vTaskDelay(pdMS_TO_TICKS(5));
}
