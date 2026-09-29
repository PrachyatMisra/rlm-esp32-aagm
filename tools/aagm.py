"""Compact RLM with the Asynchronous Adaptive Gating Mechanism (AAGM).

`CompactRLM` is a faithful, deployment-sized re-implementation of the repo's
RecursiveLanguageModel (weight-shared recursive block + learned halting gate,
soft gate mixing, masked output pooling). Two deliberate deltas vs the
notebook block, required for bit-sane MCU deployment:

  1. Gate pooling is *masked* (notebook used the full-sequence mean, which
     leaked pad-token activations into the halting decision).
  2. Inference may run with the AAGM speculative rule (below).

Halting is ACT-style (Graves, 2016): the per-step halting probability is
``p_k = 1 - g_k`` and the model halts after step ``k >= min_recursion-1``
once the cumulative mass ``sum_j p_j >= 1 - eps``. The soft gate ``g_k`` still
drives state mixing exactly like the repo model, but recursion now genuinely
terminates instead of running all ``max_recursion`` steps.

AAGM (novel mechanism, software side)
-------------------------------------
The per-step gate ``g_k`` is always honoured exactly (state mixing and ACT
halting mass are *never* approximated). What AAGM adds on top of ACT is a
tiny single-layer **shadow gate** ``s_k = sigmoid(w . pooled(h'_{k-1}))``
that forecasts, one step ahead, whether the recursion will still be running
after the next authoritative decision:

  * ``s_k < TAU_LO``  (confident halt ahead)  -> the scheduler may treat this
    sample as *retirable*. In batched serving the sample is finished after
    the current step and its HaltInfo is recorded; on the MCU the arbiter
    pre-arms the output path (head evaluation starts the moment step k's
    state is final instead of after the halting verdict, hiding head
    latency). The hint never skips or alters an authoritative step.
  * ``s_k >= TAU_LO`` (continue likely)       -> nothing changes.

Every step therefore executes identically with or without the shadow gate:
predictions, logits, depths and halting steps are *provably* invariant, while
the shadow still yields measurable scheduling wins (early batch retirement /
pre-armed output path / async bookkeeping). Calibration picks ``TAU_LO`` with
an opt-in precision floor (default 97% of retire hints are correct), and the
fraction of genuinely-early retirements is reported as ``prearm_rate``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# FNV-1a hash tokenizer (byte-identical to firmware/src/tokenizer.cpp)
# --------------------------------------------------------------------------

PAD_ID = 0
_TOKEN_RE = re.compile(rb"[a-z0-9']+")
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a_hash(token: bytes) -> int:
    h = FNV_OFFSET
    for b in token:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFF
    return h


def tokenize(text: str, max_len: int, vocab_rows: int = 4096) -> List[int]:
    """Deterministic MCU-compatible token ids (1..vocab_rows-1), padded."""
    raw = _TOKEN_RE.findall(text.encode("utf-8", "ignore").lower())
    mod = vocab_rows - 1
    ids = [fnv1a_hash(tok) % mod + 1 for tok in raw]
    ids = ids[:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


# --------------------------------------------------------------------------
# Model (architecture mirrors experiment_runner.RecursiveLanguageModel)
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_rows: int = 4096        # ids 0..4095 (row 0 = PAD); 4095 buckets
    hidden_dim: int = 96          # -> ~4x fewer hash collisions than 2047
    max_seq_len: int = 96
    num_classes: int = 2
    num_heads: int = 4
    max_recursion: int = 8
    min_recursion: int = 2
    halt_eps: float = 0.1         # ACT remainder: halt when sum(1-g) >= 1-eps
    dropout: float = 0.1

    @property
    def ffn_dim(self) -> int: return self.hidden_dim * 4
    @property
    def gate_dim(self) -> int: return self.hidden_dim // 2


class RecursiveBlock(nn.Module):
    """Weight-shared recursive step; identical op order to the repo block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(
            d, cfg.num_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim), nn.GELU(),
            nn.Linear(cfg.ffn_dim, d))
        self.gate_network = nn.Sequential(
            nn.Linear(d, cfg.gate_dim), nn.Tanh(), nn.Linear(cfg.gate_dim, 1))
        self.state_refinement = nn.Linear(d, d)
        self.cfg = cfg

    def forward(self, h, mask):
        n_real = mask.sum(dim=1)                              # [B]
        bool_pad = ~mask.bool()
        h_n = self.layer_norm1(h)
        attn_out, _ = self.attention(h_n, h_n, h_n, key_padding_mask=bool_pad)
        h = h + attn_out
        h = h + self.ffn(self.layer_norm2(h))
        pooled = (h * mask.unsqueeze(-1)).sum(1) / n_real.clamp(min=1).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_network(pooled))       # [B,1]
        h = h + 0.1 * self.state_refinement(h)
        return h, gate.squeeze(-1), pooled                    # pooled: pre-refine


class CompactRLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.embedding = nn.Embedding(cfg.vocab_rows, d, padding_idx=PAD_ID)
        self.register_buffer("pe", self._build_pe(cfg), persistent=False)
        self.block = RecursiveBlock(cfg)
        self.output_norm = nn.LayerNorm(d)
        self.classifier = nn.Sequential(
            nn.Linear(d, cfg.gate_dim), nn.GELU(), nn.Linear(cfg.gate_dim, cfg.num_classes))

    @staticmethod
    def _build_pe(cfg: ModelConfig) -> torch.Tensor:
        pe = torch.zeros(cfg.max_seq_len, cfg.hidden_dim)
        pos = torch.arange(cfg.max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, cfg.hidden_dim, 2).float() *
                        -(math.log(10000.0) / cfg.hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)                                # [1, L, D]


class ShadowGate(nn.Module):
    """One-step-ahead forecast of the authoritative halting gate."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, pooled_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(pooled_prev)).squeeze(-1)   # [B]


class AagmRLM(nn.Module):
    """CompactRLM + shadow gate + the speculative AAGM execution rule."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.core = CompactRLM(cfg)
        self.shadow = ShadowGate(cfg)
        self.tau_lo: float = 0.05     # calibrated after training
        self.cfg = cfg

    # -- embedding -------------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.core.embedding(ids)
        return h + self.core.pe[:, : ids.size(1), :]

    # -- full (non-speculative) recursion, one batch ---------------------
    def forward(self, ids, mask, budget: Optional[int] = None):
        """Returns (logits, trace dict). `budget` = runtime recursion cap
        (hardware energy-gated budget); defaults to cfg.max_recursion."""
        cfg = self.cfg
        max_rec = min(budget or cfg.max_recursion, cfg.max_recursion)
        B = ids.size(0)
        dev = ids.device
        h = self._embed(ids)
        n_real = mask.sum(1).clamp(min=1).unsqueeze(-1)

        m = mask.unsqueeze(-1).float()
        pooled_prev = (h * m).sum(1) / n_real                  # shadow input for k=0
        halted = torch.zeros(B, dtype=torch.bool, device=dev)
        depths = torch.zeros(B, device=dev)
        steps_run = torch.zeros(B, dtype=torch.long, device=dev)
        halting_mass = torch.zeros(B, device=dev)              # ACT: sum of 1-g
        gates_used, shadows, would_halt = [], [], []
        retire_hints = torch.zeros(B, dtype=torch.long, device=dev)

        for k in range(max_rec):
            steps_run += (~halted).long()
            s_k = self.shadow(pooled_prev)                     # halt forecast
            shadows.append(s_k.detach())
            h_new, g_full, pooled = self.core.block(h, mask)

            used_g = g_full
            retire_hints += ((s_k < self.tau_lo) & ~halted).long()

            running = ~halted
            g_col = used_g.unsqueeze(-1).unsqueeze(-1)
            h_next = g_col * h_new + (1.0 - g_col) * h
            h = torch.where(running.unsqueeze(-1).unsqueeze(-1), h_next, h)
            depths += running.float() * used_g
            halting_mass += running.float() * (1.0 - used_g)
            gates_used.append(used_g.detach())

            if k >= cfg.min_recursion - 1:
                newly = (halting_mass >= 1.0 - cfg.halt_eps) & running
            else:
                newly = torch.zeros_like(halted)
            would_halt.append(newly.detach())
            halted = halted | newly
            if bool(halted.all()):
                break
            pooled_prev = torch.where(running.unsqueeze(-1), pooled, pooled_prev)

        h_n = self.core.output_norm(h)
        pooled_out = (h_n * m).sum(1) / n_real
        logits = self.core.classifier(pooled_out)
        trace = {
            "depth": depths,                       # effective depth (sum of gates)
            "gates": torch.stack(gates_used),      # [steps, B]
            "shadows": torch.stack(shadows),       # [steps, B]
            "would_halt": torch.stack(would_halt), # [steps, B] authoritative halts
            "retire_hints": retire_hints,          # shadow pre-arm signals
            "steps": len(gates_used),
            "steps_run": steps_run,                # per-sample executed steps
            "ponder": depths.mean() / max_rec,     # differentiable soft-depth cost
        }
        return logits, trace
