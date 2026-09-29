"""Train the compact AAGM-RLM, calibrate the shadow gate, quantize to int8
and export everything the firmware + parity harness need.

Usage:
    python train_export.py [--dataset auto|hf|synthetic] [--epochs N]
                           [--shadow-epochs N] [--fast]

Outputs:
    esp32/firmware/rlm_esp32/src/rlm_config.h    model dims + thresholds
    esp32/firmware/rlm_esp32/src/rlm_weights.h   int8 weights + f32 params
    esp32/tools/out/golden_vectors.txt           parity vectors (int8 mirror)
    artifacts/esp32_software_results.json        full experiment record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ESP32_ROOT = HERE.parent
REPO_ROOT = ESP32_ROOT.parent
sys.path.insert(0, str(HERE))

from aagm import (AagmRLM, ModelConfig, PAD_ID, tokenize)            # noqa: E402
from corpus import build_corpus                                       # noqa: E402
from quantize import QuantTable, dequantize, quantize_per_channel     # noqa: E402

FW_SRC = ESP32_ROOT / "firmware" / "rlm_esp32" / "src"
OUT_DIR = HERE / "out"
ART_DIR = REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IdsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], max_len: int,
                 vocab_rows: int = 4096):
        self.ids = [tokenize(t, max_len, vocab_rows) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        ids = torch.tensor(self.ids[i], dtype=torch.long)
        mask = (ids != PAD_ID).long()
        return ids, mask, torch.tensor(self.labels[i], dtype=torch.long)


def collate(batch):
    ids, mask, y = zip(*batch)
    return torch.stack(ids), torch.stack(mask), torch.stack(y)


def f1_score(pred: List[int], gold: List[int]) -> float:
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


@torch.no_grad()
def evaluate(model: AagmRLM, loader: DataLoader,
             budget: int | None = None) -> Dict:
    model.eval()
    preds, golds = [], []
    depth_sum, hint_sum, steps_sum = 0.0, 0.0, 0.0
    for ids, mask, y in loader:
        logits, tr = model(ids, mask, budget=budget)
        preds += logits.argmax(1).tolist()
        golds += y.tolist()
        depth_sum += float(tr["depth"].sum())
        hint_sum += float(tr["retire_hints"].sum())
        steps_sum += float(tr["steps_run"].sum())
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n
    m_steps = steps_sum / n
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1_score(preds, golds), 4),
        "mean_depth": round(depth_sum / n, 4),
        "mean_steps": round(m_steps, 4),
        "prearm_rate": round(hint_sum / max(steps_sum, 1.0), 4),
    }


def gate_pairs(model: AagmRLM, loader: DataLoader):
    """Collect (shadow forecast at step start, halted-after-this-step) for all
    executed steps - the pairs the retire-hint threshold is calibrated on."""
    model.eval()
    s_out, halt_out = [], []
    with torch.no_grad():
        for ids, mask, _ in loader:
            _, tr = model(ids, mask)
            steps = tr["gates"].size(0)
            executed = (torch.arange(steps).unsqueeze(1)
                        < tr["steps_run"].unsqueeze(0))        # [steps, B]
            s_out.append(tr["shadows"][executed])
            halt_out.append(tr["would_halt"][executed])
    return torch.cat(s_out), torch.cat(halt_out)


def calibrate_tau_lo(model: AagmRLM, loader: DataLoader,
                     min_precision: float = 0.97) -> Dict:
    """A retire hint fires when s_k < TAU_LO and is *correct* when the
    authoritative gate halts right after that step. Pick the largest TAU_LO
    whose hints are at least `min_precision` correct (hint rate falls off
    monotonically as TAU_LO shrinks)."""
    s, halted_step = gate_pairs(model, loader)
    chosen, sweep = None, []
    for tau in torch.arange(0.001, 0.500, 0.002).tolist():
        hints = s < tau
        if not hints.any():
            continue
        prec = halted_step[hints].float().mean().item()
        row = {"tau_lo": round(tau, 3), "hint_rate": round(hints.float().mean().item(), 4),
               "precision": round(prec, 5)}
        sweep.append(row)
        if prec >= min_precision:
            chosen = row                       # largest tau meeting the bar
    if chosen is None:                         # shadow not trustworthy -> off
        chosen = {"tau_lo": 0.0, "hint_rate": 0.0,
                  "precision": sweep[-1]["precision"] if sweep else None}
    return {"tau_lo": chosen["tau_lo"], "chosen": chosen, "sweep": sweep}


@torch.no_grad()
def build_quant_table(model: AagmRLM) -> QuantTable:
    t = QuantTable()
    core, blk = model.core, model.core.block

    def lin(name: str, m: nn.Linear):
        t.add_linear(name, m.weight, m.bias)

    t.add_linear("emb", core.embedding.weight)
    t.add_linear("attn_in", blk.attention.in_proj_weight, blk.attention.in_proj_bias)
    lin("attn_out", blk.attention.out_proj)
    lin("ffn1", blk.ffn[0])
    lin("ffn2", blk.ffn[2])
    lin("gate1", blk.gate_network[0])
    lin("gate2", blk.gate_network[2])
    lin("refine", blk.state_refinement)
    lin("cls1", core.classifier[0])
    lin("cls2", core.classifier[2])
    lin("shadow", model.shadow.proj)
    t.add_f32("ln1_w", blk.layer_norm1.weight)
    t.add_f32("ln1_b", blk.layer_norm1.bias)
    t.add_f32("ln2_w", blk.layer_norm2.weight)
    t.add_f32("ln2_b", blk.layer_norm2.bias)
    t.add_f32("lnout_w", core.output_norm.weight)
    t.add_f32("lnout_b", core.output_norm.bias)
    return t


@torch.no_grad()
def dequant_mirror(model: AagmRLM, table: QuantTable) -> AagmRLM:
    """Firmware-equivalent float copy: every int8 weight round-tripped."""
    mirror = AagmRLM(model.cfg)
    mirror.load_state_dict(model.state_dict())
    mirror.tau_lo = model.tau_lo
    core, blk, t = mirror.core, mirror.core.block, table

    def patch(name, param):
        param.copy_(dequantize(t.q[name], t.scale[name]))

    patch("emb", core.embedding.weight)
    patch("attn_in", blk.attention.in_proj_weight)
    patch("attn_out", blk.attention.out_proj.weight)
    patch("ffn1", blk.ffn[0].weight)
    patch("ffn2", blk.ffn[2].weight)
    patch("gate1", blk.gate_network[0].weight)
    patch("gate2", blk.gate_network[2].weight)
    patch("refine", blk.state_refinement.weight)
    patch("cls1", core.classifier[0].weight)
    patch("cls2", core.classifier[2].weight)
    patch("shadow", mirror.shadow.proj.weight)
    return mirror


@torch.no_grad()
def probe_model(model: AagmRLM, cfg: ModelConfig):
    """Out-of-template probes: single-word sentiment grounding over the full
    generator lexicon + the curated GOLDEN sentences."""
    from corpus import LEXICON_PAIRS
    model.eval()

    def pred_of(text: str) -> int:
        ids = torch.tensor([tokenize(text, cfg.max_seq_len, cfg.vocab_rows)])
        mask = (ids != PAD_ID).long()
        logits, _ = model(ids, mask)
        return int(logits.argmax(1))

    ground = sum(pred_of(f"this film is {w}") == y for w, y in LEXICON_PAIRS)
    curated = sum(pred_of(t) == y for t, y in GOLDEN_SAMPLES)
    return (round(ground / len(LEXICON_PAIRS), 4),
            round(curated / len(GOLDEN_SAMPLES), 4))


# --------------------------------------------------------------------------
def c_ident(name: str) -> str:
    return name.replace(".", "_")


def emit_headers(cfg: ModelConfig, table: QuantTable, tau_lo: float) -> None:
    FW_SRC.mkdir(parents=True, exist_ok=True)
    c = cfg
    config = f"""// AUTO-GENERATED by esp32/tools/train_export.py - do not edit.
#ifndef RLM_CONFIG_H
#define RLM_CONFIG_H

#define RLM_VOCAB_ROWS   {c.vocab_rows}
#define RLM_DIM          {c.hidden_dim}
#define RLM_MAX_SEQ      {c.max_seq_len}
#define RLM_HEADS        {c.num_heads}
#define RLM_HEAD_DIM     (RLM_DIM / RLM_HEADS)
#define RLM_FFN          {c.ffn_dim}
#define RLM_GATE_HID     {c.gate_dim}
#define RLM_CLASSES      {c.num_classes}
#define RLM_MAX_RECURSION {c.max_recursion}
#define RLM_MIN_RECURSION {c.min_recursion}
#define RLM_HALT_EPS {c.halt_eps}f
#define RLM_TAU_LO         {tau_lo}f
#define RLM_LN_EPS       1e-5f
#define RLM_REFINE_GAIN  0.1f

#endif  // RLM_CONFIG_H
"""
    (FW_SRC / "rlm_config.h").write_text(config)

    lines: List[str] = [
        "// AUTO-GENERATED by esp32/tools/train_export.py - do not edit.",
        "// Symmetric per-channel int8 weights (zero_point = 0) + f32 params.",
        "#ifndef RLM_WEIGHTS_H",
        "#define RLM_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        '#include "rlm_config.h"',
        "",
    ]

    def emit_q(name: str) -> None:
        q, s = table.q[name], table.scale[name]
        rows, cols = q.shape
        vals = q.flatten().tolist()
        body = ",".join(str(v) for v in vals)
        lines.append(f"static const int8_t w_{c_ident(name)}[{rows}][{cols}] = {{{body}}};")
        svals = ",".join(f"{v:.9g}f" for v in s.tolist())
        lines.append(f"static const float s_{c_ident(name)}[{rows}] = {{{svals}}};")

    def emit_f(name: str) -> None:
        v = table.f32[name].flatten().tolist()
        body = ",".join(f"{x:.9g}f" for x in v)
        lines.append(f"static const float p_{c_ident(name)}[{len(v)}] = {{{body}}};")

    for n in ["emb", "attn_in", "attn_out", "ffn1", "ffn2",
              "gate1", "gate2", "refine", "cls1", "cls2", "shadow"]:
        emit_q(n)
    for n in table.f32:
        emit_f(n)

    lines.append("")
    lines.append("#endif  // RLM_WEIGHTS_H")
    (FW_SRC / "rlm_weights.h").write_text("\n".join(lines))


# --------------------------------------------------------------------------
def emit_golden(model: AagmRLM, samples: List[tuple]) -> Path:
    """Line-pair format: 'TEXT <raw>' then a whitespace row of numbers:
    pred logit0 logit1 depth steps n_tokens ids[0..95] gates[0..7(-1 pad)]
    shadows[0..7(-1 pad)]"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "golden_vectors.txt"
    model.eval()
    rows: List[str] = []
    with torch.no_grad():
        for text, _expected in samples:
            ids = tokenize(text, model.cfg.max_seq_len, model.cfg.vocab_rows)
            t_ids = torch.tensor([ids])
            mask = (t_ids != PAD_ID).long()
            logits, tr = model(t_ids, mask)
            pred = int(logits.argmax(1))
            n_real = int(mask.sum())
            gates = tr["gates"][:, 0].tolist()
            shadows = tr["shadows"][:, 0].tolist()
            gates += [-1.0] * (model.cfg.max_recursion - len(gates))
            shadows += [-1.0] * (model.cfg.max_recursion - len(shadows))
            nums = [pred, logits[0, 0].item(), logits[0, 1].item(),
                    tr["depth"][0].item(), tr["steps"], n_real] + ids \
                + gates + shadows
            rows.append("TEXT " + text)
            rows.append(" ".join(f"{x:.6f}" if isinstance(x, float) else str(x)
                                 for x in nums))
    path.write_text("\n".join(rows) + "\n")
    return path


# Curated natural-language probes (text, expected label) used both for the
# parity vectors and for an out-of-template sanity metric.
GOLDEN_SAMPLES = [
    ("the movie was a brilliant masterpiece with stunning visuals and a powerful story.", 1),
    ("an absolute disaster, boring and painfully slow from start to finish.", 0),
    ("it starts slow but the brilliant ending makes it absolutely worth watching.", 1),
    ("great cast but the terrible script ruins the whole film completely.", 0),
    ("this film is not bad at all, i actually enjoyed every minute of it.", 1),
    ("the acting is good yet the plot feels dull, hollow and predictable.", 0),
    ("a wonderful charming little film, smart and genuinely funny throughout.", 1),
    ("i wanted to love it, but sadly the movie is a tedious lifeless mess.", 0),
    ("short review: superb.", 1),
    ("bad.", 0),
    ("the first act is clumsy and uneven meanwhile the cast shines, some scenes run slow, yet the final scene feels magnificent.", 1),
    ("watched it last night and honestly it was neither great nor terrible but the ending is weak so i cannot recommend it.", 0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto", choices=["auto", "hf", "synthetic"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--shadow-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ponder", type=float, default=0.02,
                    help="ACT-style weight on the differentiable ponder cost")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="tiny corpus for smoke runs")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    set_seed(args.seed)
    source = "hf" if args.dataset == "hf" else "synthetic" if args.dataset == "synthetic" else "hf"
    n_tr, n_val, n_te = (400, 120, 200) if args.fast else (1600, 400, 600)
    t0 = time.time()
    corpus = build_corpus(source, args.seed, n_tr, n_val, n_te)

    cfg = ModelConfig()
    model = AagmRLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    def loader(texts, labels, bs, shuffle):
        return DataLoader(IdsDataset(texts, labels, cfg.max_seq_len, cfg.vocab_rows),
                          batch_size=bs, shuffle=shuffle, collate_fn=collate)

    tr_l = loader(corpus.train_texts, corpus.train_labels, args.batch, True)
    va_l = loader(corpus.val_texts, corpus.val_labels, args.batch, False)
    te_l = loader(corpus.test_texts, corpus.test_labels, args.batch, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    history = []
    print(f"[train] phase 1: full-gate + ponder training, {n_params:,} params")
    for ep in range(args.epochs):
        model.train()
        tot, run_n = 0.0, 0
        for ids, mask, y in tr_l:
            opt.zero_grad()
            logits, tr = model(ids, mask)
            loss = ce(logits, y) + args.ponder * tr["ponder"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * ids.size(0)
            run_n += ids.size(0)
        row = {"epoch": ep, "loss": round(tot / run_n, 5),
               **{f"val_{k}": v for k, v in evaluate(model, va_l).items()}}
        history.append(row)
        print(f"  ep{ep}: loss={row['loss']:.4f} val_acc={row['val_accuracy']:.4f}")

    calib = calibrate_tau_lo(model, tr_l)
    model.tau_lo = calib["tau_lo"]
    print(f"[calib] TAU_LO={model.tau_lo} "
          f"(precision={calib['chosen']['precision']}, "
          f"hint_rate={calib['chosen']['hint_rate']})")

    bce = nn.BCELoss(reduction="none")
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
    print(f"[train] phase 2: shadow-hint fine-tune (tau_lo={model.tau_lo})")
    for ep in range(args.shadow_epochs):
        model.train()
        tot = 0.0
        run_n = 0
        for ids, mask, y in tr_l:
            opt2.zero_grad()
            logits, tr = model(ids, mask)
            steps = tr["gates"].size(0)
            valid = (torch.arange(steps).unsqueeze(1)
                     < tr["steps_run"].unsqueeze(0)).float()   # executed steps
            target = tr["would_halt"].float()  # shadow learns to *predict halts*
            shadow_loss = (bce(tr["shadows"].clamp(1e-6, 1 - 1e-6), target.detach())
                           * valid).sum() / valid.sum().clamp(min=1.0)
            loss = ce(logits, y) + 0.3 * shadow_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            tot += loss.item() * ids.size(0)
            run_n += ids.size(0)
        print(f"  shadow ep{ep}: loss={tot / run_n:.4f}")

    calib2 = calibrate_tau_lo(model, tr_l)
    model.tau_lo = calib2["tau_lo"]
    print(f"[calib] post-finetune TAU_LO={model.tau_lo} "
          f"(precision={calib2['chosen']['precision']}, "
          f"hint_rate={calib2['chosen']['hint_rate']})")

    res_full = evaluate(model, te_l)
    # fixed-depth ablation: halting mass never reaches threshold -> always 8
    save_min, save_eps = cfg.min_recursion, cfg.halt_eps
    cfg.min_recursion = cfg.max_recursion
    cfg.halt_eps = -1e9
    res_fix8 = evaluate(model, te_l)
    cfg.min_recursion, cfg.halt_eps = save_min, save_eps
    res_fix8["mean_depth"] = res_fix8["mean_steps"]
    # hardware (energy-context) depth-budget sweep: what the ECO profiles cost
    budget_sweep = {str(b): evaluate(model, te_l, budget=b)
                    for b in (4, 6, 8)}

    table = build_quant_table(model)
    mirror = dequant_mirror(model, table)
    res_mirror = evaluate(mirror, te_l)
    grounding, curated_ok = probe_model(mirror, cfg)
    emit_headers(cfg, table, model.tau_lo)
    emit_golden(mirror, GOLDEN_SAMPLES)

    # per-position MACs of one recursive step (attention keys excluded):
    # qkv 3d^2 + attn weighted-sum ~ L*d (per pos) + out d^2 + ffn 2*d*4d + refine d^2
    d = cfg.hidden_dim
    mac_per_step = cfg.max_seq_len * (3 * d * d + cfg.max_seq_len * d
                                      + d * d + 2 * d * d * 4 + d * d)
    record = {
        "config": {
            "hidden_dim": cfg.hidden_dim, "max_seq_len": cfg.max_seq_len,
            "heads": cfg.num_heads, "vocab_rows": cfg.vocab_rows,
            "max_recursion": cfg.max_recursion, "min_recursion": cfg.min_recursion,
            "halt_eps": cfg.halt_eps, "tau_lo": model.tau_lo,
            "params_float": n_params, "int8_bytes": table.int8_bytes,
            "f32_param_bytes": table.float_bytes, "dataset": source,
        },
        "phase1_history": history,
        "calibration": {"phase1": calib, "final": calib2},
        "test_aagm": res_full,
        "test_fixed_depth8": res_fix8,
        "test_int8_mirror_aagm": res_mirror,
        "mirror_lexicon_grounding": grounding,
        "mirror_curated_accuracy": curated_ok,
        "test_by_budget": budget_sweep,
        "approx_flops_mac_per_recursion_step": mac_per_step,
        "train_seconds": round(time.time() - t0, 1),
    }
    ART_DIR.mkdir(exist_ok=True)
    (ART_DIR / "esp32_software_results.json").write_text(json.dumps(record, indent=2))
    print("[done] aagm:", res_full)
    print("[done] fixed8:", res_fix8)
    print("[done] int8 mirror:", res_mirror)
    print(f"[done] mirror probes: lexicon={grounding} curated={curated_ok}")
    print(f"[done] int8 weights: {table.int8_bytes/1024:.0f} KiB + "
          f"{table.float_bytes/1024:.0f} KiB f32 params")


if __name__ == "__main__":
    main()
