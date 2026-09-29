/* tokenizer.cpp - FNV-1a hash tokenizer, byte-identical to
 * tools/aagm.py:tokenize. Words are maximal runs of [a-z0-9'] after ASCII
 * lowercasing; every other byte (including UTF-8 continuation bytes) is a
 * separator on both sides, so host and firmware always agree. */
#include "rlm_engine.h"

#define FNV_OFFSET 2166136261u
#define FNV_PRIME  16777619u
#define VOCAB_IDS  (RLM_VOCAB_ROWS - 1)   /* usable ids 1..ROWS-1, 0 = PAD */

static inline uint8_t is_tok_char(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '\'';
}

uint8_t rlm_tokenize(const char* text, uint16_t* ids) {
    uint8_t n = 0;
    uint32_t h = FNV_OFFSET;
    uint8_t in_tok = 0;
    for (const uint8_t* p = (const uint8_t*)text; ; ++p) {
        uint8_t c = *p;
        if (c >= 'A' && c <= 'Z') c = (uint8_t)(c | 0x20);
        if (is_tok_char(c)) {
            h = (h ^ c) * FNV_PRIME;
            in_tok = 1;
        } else {
            if (in_tok && n < RLM_MAX_SEQ)
                ids[n++] = (uint16_t)(h % VOCAB_IDS + 1);
            in_tok = 0;
            h = FNV_OFFSET;
            if (c == 0 || n >= RLM_MAX_SEQ) break;
        }
    }
    while (n < RLM_MAX_SEQ) ids[n++] = 0;   /* zero-pad */
    uint8_t real = 0;
    while (real < RLM_MAX_SEQ && ids[real] != 0) ++real;
    return real;
}
