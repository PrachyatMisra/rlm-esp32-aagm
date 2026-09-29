/* host_harness.cpp - native (g++) parity harness.
 *
 * Compiles the *actual* firmware engine (rlm_engine.cpp + tokenizer.cpp)
 * on the host and compares it against golden vectors produced by the PyTorch
 * int8-dequant mirror (tools/train_export.py -> out/golden_vectors.txt).
 * Passing this proves the firmware C++ is functionally equal to the model
 * that was trained and quantized in software, at decision granularity
 * (exact same steps/pred) and float tolerance (logits/gates).
 *
 *   ./host_harness <golden_vectors.txt> [--bench N]
 *
 * Emits one JSON report to stdout.
 */
#include "../rlm_esp32/src/rlm_engine.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

static const float LOGIT_ATOL = 2e-2f;
static const float GATE_ATOL  = 1e-3f;
static const float DEPTH_ATOL = 5e-3f;

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
    if (argc < 2) { fprintf(stderr, "usage: %s golden.txt [--bench N]\n", argv[0]); return 2; }
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
