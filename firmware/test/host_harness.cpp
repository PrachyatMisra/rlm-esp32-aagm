/* host_harness.cpp - native (g++) parity harness & hardware-free simulator.
 *
 * Compiles the *actual* firmware engine (rlm_engine.cpp + tokenizer.cpp)
 * on the host. Three execution modes:
 *
 *   1. Golden Parity (automated regression):
 *      ./host_harness <golden_vectors.txt> [--bench N]
 *      Emits JSON parity report vs PyTorch int8 mirror expectations.
 *
 *   2. Single Shot Inference:
 *      ./host_harness --infer "<text>" [--budget N] [--profile PERF|BAL|ECO]
 *      Runs tokenizer and engine, emitting device-identical telemetry JSON.
 *
 *   3. Interactive Protocol (Hardware-Free Mock UART):
 *      ./host_harness --interactive
 *      Speaks the EXACT line-oriented FreeRTOS serial protocol of rlm_esp32.ino
 *      over stdin/stdout (INFER, CHAT, MODE, BATT, BENCH, STAT, PING, ABORT).
 */
#include "../rlm_esp32/src/rlm_engine.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

static const float LOGIT_ATOL = 2e-2f;
static const float GATE_ATOL  = 1e-3f;
static const float DEPTH_ATOL = 5e-3f;

/* ---------------- Profile & Simulated Hardware State ------------------ */
enum Profile : uint8_t { PROF_PERF = 0, PROF_BAL = 1, PROF_ECO = 2 };
static const uint8_t  PROF_BUDGET[3] = { 8, 6, 4 };
static const uint32_t PROF_MHZ[3]    = { 240, 160, 80 };
static const char*    PROF_NAME[3]   = { "PERF", "BAL", "ECO" };

static volatile uint8_t  g_profile   = PROF_PERF;
static volatile bool     g_auto_mode = false;
static volatile uint8_t  g_budget    = RLM_MAX_RECURSION;
static volatile int      g_abort     = 0;
static volatile bool     g_batt_auto = false;
static float             g_batt_mv   = 4000.0f; /* OCV */
static const char*       g_batt_src  = "sim";
static uint32_t          g_uptime_ms = 0;

static const float BATT_R_INT = 0.200f; // 200 mOhm internal resistance

static float compute_loaded_mv(float ocv_mv, uint32_t cpu_mhz) {
    float i_ma = (cpu_mhz >= 240) ? 50.0f : ((cpu_mhz >= 160) ? 36.0f : 27.0f);
    return ocv_mv - (i_ma / 1000.0f) * BATT_R_INT * 1000.0f;
}

static int fw_sim_abort(void*) { return g_abort != 0; }

static RlmGateProvider FW_SIM_PROVIDER = {
    RLM_LOCAL_PROVIDER.prime_shadow,
    RLM_LOCAL_PROVIDER.launch_gate,
    RLM_LOCAL_PROVIDER.wait_gate,
    RLM_LOCAL_PROVIDER.wait_shadow,
    fw_sim_abort,
    nullptr
};

static void apply_profile(uint8_t prof) {
    if (prof > 2) prof = 2;
    uint8_t new_budget = PROF_BUDGET[prof];
    if (new_budget < g_budget) g_abort = 1;     /* budget shrink triggers abort */
    g_profile = prof;
    g_budget  = new_budget;
}

static uint8_t profile_for_batt(float loaded_mv) {
    float pct = (loaded_mv - 3300.0f) / 900.0f * 100.0f;
    if (pct < 0.0f) pct = 0.0f;
    if (pct > 100.0f) pct = 100.0f;
    if (pct < 25.0f) return PROF_ECO;
    if (pct < 50.0f) return PROF_BAL;
    return PROF_PERF;
}

static void energy_tick() {
    float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
    if (loaded < 3350.0f) {
        apply_profile(PROF_ECO);
        g_abort = 1; // Brownout mitigation trigger
    } else if (g_auto_mode) {
        apply_profile(profile_for_batt(loaded));
    }
}

static void emit_infer_json(const char* text, uint8_t budget) {
    uint16_t ids[RLM_MAX_SEQ];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        printf("{\"err\":\"empty\"}\n");
        fflush(stdout);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    RlmResult r;
    rlm_engine_infer(ids, n_real, budget, &FW_SIM_PROVIDER, &r);
    auto t1 = std::chrono::steady_clock::now();
    uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();

    float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
    float droop = g_batt_mv - loaded;

    printf("{\"pred\":%u,\"logits\":[%.4f,%.4f],\"steps\":%u,"
           "\"depth\":%.4f,\"mass\":%.4f,\"hints\":%u,\"aborted\":%u,"
           "\"spec_hits\":%u,\"frozen\":%u,\"macs_saved\":%u,\"tail_us_saved\":%u,"
           "\"gates\":[", r.pred, r.logits[0], r.logits[1],
           r.trace.steps, r.trace.depth, r.trace.halting_mass,
           r.trace.prearm_hints, r.trace.aborted,
           r.trace.speculative_head_hits, r.trace.frozen_tokens,
           (unsigned)r.trace.ffn_macs_saved, (unsigned)r.trace.tail_latency_us_saved);
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        printf(i ? ",%.4f" : "%.4f", r.trace.gates[i]);
    printf("],\"shadows\":[");
    for (uint8_t i = 0; i < r.trace.steps; ++i)
        printf(i ? ",%.4f" : "%.4f", r.trace.shadows[i]);
    printf("],\"us\":%u,\"budget\":%u,\"profile\":\"%s\",\"batt_mv\":%.0f,"
           "\"loaded_mv\":%.0f,\"droop_mv\":%.1f,\"cpu_mhz\":%u,\"heap\":284120}\n",
           us, budget, PROF_NAME[g_profile], (double)g_batt_mv,
           (double)loaded, (double)droop, (unsigned)PROF_MHZ[g_profile]);
    fflush(stdout);
}

static void emit_chat_json(const char* text) {
    uint16_t ids[RLM_MAX_SEQ];
    uint8_t n_real = rlm_tokenize(text, ids);
    if (n_real == 0) {
        printf("{\"chat_reply\":\"Please enter a valid message.\"}\n");
        fflush(stdout);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    RlmResult r;
    rlm_engine_infer(ids, n_real, g_budget, &FW_SIM_PROVIDER, &r);
    auto t1 = std::chrono::steady_clock::now();
    uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();

    float max_l = std::max(r.logits[0], r.logits[1]);
    float e0 = expf(r.logits[0] - max_l);
    float e1 = expf(r.logits[1] - max_l);
    float conf = (r.pred == 1 ? e1 : e0) / (e0 + e1) * 100.0f;
    const char* verdict = r.pred == 1 ? "POSITIVE" : "NEGATIVE";

    printf("{\"chat_reply\":\"I processed: '%s'. Verdict: %s (Confidence: %.1f%%). "
           "Recursive depth: %u steps, cumulative mass: %.3f/0.900, "
           "saving %u steps (%.0f%% compute reduction). Latency: %u us.\","
           "\"pred\":%u,\"verdict\":\"%s\",\"confidence\":%.1f,"
           "\"steps\":%u,\"saved_steps\":%u,\"us\":%u,\"profile\":\"%s\"}\n",
           text, verdict, conf, r.trace.steps, r.trace.halting_mass,
           (8 - r.trace.steps), (float)(8 - r.trace.steps) / 8.0f * 100.0f,
           us, r.pred, verdict, conf, r.trace.steps,
           (8 - r.trace.steps), us, PROF_NAME[g_profile]);
    fflush(stdout);
}

static void run_interactive() {
    rlm_engine_init();
    printf(">RLM-AAGM fw=2 board=ESP32-DevKit-v1(HostSim) vocab=%u dim=%u max_rec=%u "
           "tau_lo=%.3f budget=%u chat=ready\n", RLM_VOCAB_ROWS, RLM_DIM, RLM_MAX_RECURSION,
           (double)RLM_TAU_LO, g_budget);
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;

        if (line.rfind("CHAT ", 0) == 0) {
            std::string text = line.substr(5);
            g_abort = 0;
            emit_chat_json(text.c_str());
        }
        else if (line.rfind("INFER ", 0) == 0) {
            std::string text = line.substr(6);
            g_abort = 0;
            emit_infer_json(text.c_str(), g_budget);
        }
        else if (line.rfind("BENCH", 0) == 0) {
            int n = 20;
            if (line.size() > 6) n = std::max(1, atoi(line.c_str() + 6));
            const char* text = "the first act is clumsy and uneven meanwhile the cast "
                               "shines, yet the final scene feels magnificent";
            uint16_t ids[RLM_MAX_SEQ];
            uint8_t n_real = rlm_tokenize(text, ids);
            uint32_t best = UINT32_MAX, worst = 0, steps_sum = 0;
            double sum = 0;
            for (int i = 0; i < n; ++i) {
                g_abort = 0;
                auto t0 = std::chrono::steady_clock::now();
                RlmResult r;
                rlm_engine_infer(ids, n_real, g_budget, &FW_SIM_PROVIDER, &r);
                auto t1 = std::chrono::steady_clock::now();
                uint32_t us = (uint32_t)std::chrono::duration<double, std::micro>(t1 - t0).count();
                best = std::min(best, us);
                worst = std::max(worst, us);
                sum += us;
                steps_sum += r.trace.steps;
            }
            printf("{\"bench_n\":%d,\"us_avg\":%.1f,\"us_min\":%u,\"us_max\":%u,"
                   "\"mean_steps\":%.3f,\"budget\":%u,\"cpu_mhz\":%u,"
                   "\"profile\":\"%s\",\"heap\":284120}\n",
                   n, sum / n, (unsigned)best, (unsigned)worst,
                   (double)steps_sum / n, g_budget,
                   (unsigned)PROF_MHZ[g_profile], PROF_NAME[g_profile]);
            fflush(stdout);
        }
        else if (line == "STAT") {
            float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
            printf("{\"board\":\"ESP32-DevKit-v1(HostSim)\",\"cpu_mhz\":%u,\"budget\":%u,"
                   "\"profile\":\"%s\",\"auto_mode\":%s,\"batt_mv\":%.0f,"
                   "\"loaded_mv\":%.0f,\"batt_src\":\"%s\",\"heap\":284120,\"min_heap\":241088,"
                   "\"sketch_kb\":510,\"flash_mb\":4}\n",
                   (unsigned)PROF_MHZ[g_profile], g_budget,
                   PROF_NAME[g_profile], g_auto_mode ? "true" : "false",
                   (double)g_batt_mv, (double)loaded, g_batt_src);
            fflush(stdout);
        }
        else if (line == "PING") {
            g_uptime_ms += 100;
            printf("{\"pong\":1,\"uptime_ms\":%u}\n", g_uptime_ms);
            fflush(stdout);
        }
        else if (line.rfind("MODE ", 0) == 0) {
            std::string m = line.substr(5);
            if      (m == "PERF") { g_auto_mode = false; apply_profile(PROF_PERF); }
            else if (m == "BAL")  { g_auto_mode = false; apply_profile(PROF_BAL); }
            else if (m == "ECO")  { g_auto_mode = false; apply_profile(PROF_ECO); }
            else if (m == "AUTO") { g_auto_mode = true; energy_tick(); }
            printf("{\"mode\":\"%s\",\"budget\":%u,\"cpu_mhz\":%u}\n",
                   PROF_NAME[g_profile], g_budget, (unsigned)PROF_MHZ[g_profile]);
            fflush(stdout);
        }
        else if (line.rfind("BATT ", 0) == 0) {
            std::string b = line.substr(5);
            if (b == "AUTO") { g_batt_auto = true; }
            else { g_batt_auto = false; g_batt_mv = (float)atof(b.c_str()); g_batt_src = "sim"; }
            energy_tick();
            float loaded = compute_loaded_mv(g_batt_mv, PROF_MHZ[g_profile]);
            printf("{\"batt_mv\":%.0f,\"loaded_mv\":%.0f,\"batt_src\":\"%s\",\"profile\":\"%s\",\"budget\":%u}\n",
                   (double)g_batt_mv, (double)loaded, g_batt_src, PROF_NAME[g_profile], g_budget);
            fflush(stdout);
        }
        else if (line.rfind("ABORT", 0) == 0) {
            g_abort = 1;
            printf("{\"abort_armed\":1}\n");
            fflush(stdout);
        }
        else if (line == "QUIT" || line == "EXIT") {
            break;
        }
        else {
            printf("{\"err\":\"unknown cmd\"}\n");
            fflush(stdout);
        }
    }
}

/* ---------------- Golden Row Parsing ---------------------------------- */
struct Golden {
    std::string text;
    int pred;
    float logits[2];
    float depth;
    int steps;
    int n_real;
    uint16_t ids[RLM_MAX_SEQ];
    float gates[RLM_MAX_RECURSION];
    float shadows[RLM_MAX_RECURSION];
};

static bool parse_row(const std::string& line, Golden& g) {
    std::vector<float> v;
    const char* p = line.c_str();
    char* end = nullptr;
    while (*p) {
        float x = strtof(p, &end);
        if (end == p) { ++p; continue; }
        v.push_back(x);
        p = end;
    }
    if (v.size() != (size_t)(6 + RLM_MAX_SEQ + 2 * RLM_MAX_RECURSION)) return false;
    size_t i = 0;
    g.pred  = (int)v[i++];
    g.logits[0] = v[i++];
    g.logits[1] = v[i++];
    g.depth = v[i++];
    g.steps = (int)v[i++];
    g.n_real = (int)v[i++];
    for (int t = 0; t < RLM_MAX_SEQ; ++t) g.ids[t] = (uint16_t)v[i++];
    for (int t = 0; t < RLM_MAX_RECURSION; ++t) g.gates[t] = v[i++];
    for (int t = 0; t < RLM_MAX_RECURSION; ++t) g.shadows[t] = v[i++];
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage:\n"
                        "  %s golden.txt [--bench N]\n"
                        "  %s --infer \"<text>\" [--budget N] [--profile PERF|BAL|ECO]\n"
                        "  %s --interactive\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }

    /* 1. Interactive mock UART loop */
    if (!strcmp(argv[1], "--interactive") || !strcmp(argv[1], "-i")) {
        run_interactive();
        return 0;
    }

    /* 2. Single shot infer */
    if (!strcmp(argv[1], "--infer")) {
        if (argc < 3) { fprintf(stderr, "error: --infer requires text\n"); return 2; }
        const char* text = argv[2];
        uint8_t budget = RLM_MAX_RECURSION;
        for (int i = 3; i + 1 < argc; ++i) {
            if (!strcmp(argv[i], "--budget")) budget = (uint8_t)atoi(argv[i + 1]);
            if (!strcmp(argv[i], "--profile")) {
                if (!strcmp(argv[i+1], "BAL")) apply_profile(PROF_BAL);
                else if (!strcmp(argv[i+1], "ECO")) apply_profile(PROF_ECO);
                else apply_profile(PROF_PERF);
                budget = g_budget;
            }
        }
        rlm_engine_init();
        emit_infer_json(text, budget);
        return 0;
    }

    /* 3. Golden verification mode */
    int bench_n = 0;
    for (int i = 2; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--bench")) bench_n = atoi(argv[i + 1]);

    FILE* f = fopen(argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::vector<Golden> gs;
    char buf[8192];
    std::string text;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.rfind("TEXT ", 0) == 0) { text = line.substr(5); continue; }
        if (line.empty()) continue;
        Golden g;
        if (parse_row(line, g)) { g.text = text; gs.push_back(g); }
    }
    fclose(f);
    if (gs.empty()) { fprintf(stderr, "no golden rows parsed\n"); return 2; }

    rlm_engine_init();

    float max_le = 0.f, max_ge = 0.f, max_de = 0.f, max_se = 0.f;
    int mism_pred = 0, mism_steps = 0, mism_ids = 0, pass = 0;
    long steps_sum = 0, hint_sum = 0;
    printf("{\"samples\":[");
    for (size_t i = 0; i < gs.size(); ++i) {
        Golden& g = gs[i];

        /* C tokenizer must reproduce the golden ids exactly */
        uint16_t ids[RLM_MAX_SEQ];
        uint8_t n_real = rlm_tokenize(g.text.c_str(), ids);
        bool ids_ok = (n_real == g.n_real);
        for (int t = 0; ids_ok && t < RLM_MAX_SEQ; ++t) ids_ok &= (ids[t] == g.ids[t]);
        if (!ids_ok) ++mism_ids;

        RlmResult r;
        rlm_engine_infer(g.ids, (uint8_t)g.n_real, RLM_MAX_RECURSION,
                         &RLM_LOCAL_PROVIDER, &r);

        float le = fmaxf(fabsf(r.logits[0] - g.logits[0]),
                         fabsf(r.logits[1] - g.logits[1]));
        float de = fabsf(r.trace.depth - g.depth);
        float ge = 0.f, se = 0.f;
        for (int k = 0; k < g.steps && k < RLM_MAX_RECURSION; ++k) {
            ge = fmaxf(ge, fabsf(r.trace.gates[k] - g.gates[k]));
            se = fmaxf(se, fabsf(r.trace.shadows[k] - g.shadows[k]));
        }
        max_le = fmaxf(max_le, le); max_de = fmaxf(max_de, de);
        max_ge = fmaxf(max_ge, ge); max_se = fmaxf(max_se, se);
        bool ok = ((int)r.pred == g.pred) && ((int)r.trace.steps == g.steps) &&
                  ids_ok && le < LOGIT_ATOL && de < DEPTH_ATOL && ge < GATE_ATOL;
        mism_pred += ((int)r.pred != g.pred);
        mism_steps += ((int)r.trace.steps != g.steps);
        pass += ok;
        steps_sum += r.trace.steps;
        hint_sum += r.trace.prearm_hints;
        printf("%s{\"i\":%zu,\"pred\":%d,\"gold_pred\":%d,\"steps\":%d,"
               "\"gold_steps\":%d,\"depth\":%.4f,\"logit_err\":%.2e,"
               "\"gate_err\":%.2e,\"shadow_err\":%.2e,\"ids_ok\":%s,\"ok\":%s}",
               i ? "," : "", i, r.pred, g.pred, r.trace.steps, g.steps,
               r.trace.depth, le, ge, se, ids_ok ? "true" : "false",
               ok ? "true" : "false");
    }
    printf("],\"summary\":{");

    /* determinism + rough host speed (bench: rerun first text N times) */
    double us_min = 1e12, us_avg = 0.0;
    if (bench_n > 0) {
        for (int b = 0; b < bench_n; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            RlmResult r;
            rlm_engine_infer(gs[0].ids, (uint8_t)gs[0].n_real, RLM_MAX_RECURSION,
                             &RLM_LOCAL_PROVIDER, &r);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            us_min = us < us_min ? us : us_min;
            us_avg += us;
        }
        us_avg /= bench_n;
    }
    printf("\"golden\":%zu,\"passed\":%d,\"max_logit_err\":%.3e,"
           "\"max_gate_err\":%.3e,\"max_shadow_err\":%.3e,\"max_depth_err\":%.3e,"
           "\"mism_pred\":%d,\"mism_steps\":%d,\"mism_ids\":%d,"
           "\"mean_steps\":%.3f,\"mean_prearm_hints\":%.3f,"
           "\"bench_us_avg\":%.1f,\"bench_us_min\":%.1f,"
           "\"PASS\":%s}\n",
           gs.size(), pass, max_le, max_ge, max_se, max_de,
           mism_pred, mism_steps, mism_ids,
           (double)steps_sum / gs.size(), (double)hint_sum / gs.size(),
           us_avg == 0.0 ? 0.0 : us_avg, us_min == 1e12 ? 0.0 : us_min,
           (pass == (int)gs.size()) ? "true" : "false");
    printf("}\n");
    return pass == (int)gs.size() ? 0 : 1;
}
