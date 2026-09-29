#!/usr/bin/env python3
"""Interactive CLI demo for the ESP32 AAGM-RLM.

Allows real-time interactive inference on arbitrary text without physical
ESP32 hardware, showcasing tokenization, ACT halting, shadow forecaster,
and energy profile comparisons.

Usage:
    python tools/demo.py
    python tools/demo.py --text "the film was breathtaking and magnificent"
    python tools/demo.py --mode ECO --budget 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HARNESS = REPO_ROOT / "firmware" / "test" / "host_harness"


def run_infer(text: str, budget: int = 8, profile: str = "PERF") -> dict:
    if not HARNESS.exists():
        subprocess.run(["make", "-C", str(HARNESS.parent), "host_harness"],
                       check=True, capture_output=True)
    cmd = [str(HARNESS), "--infer", text, "--budget", str(budget), "--profile", profile]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip())


def display_result(text: str, res: dict) -> None:
    pred = res.get("pred", 0)
    sentiment = "POSITIVE (+)" if pred == 1 else "NEGATIVE (-)"
    logits = res.get("logits", [0.0, 0.0])
    steps = res.get("steps", 0)
    budget = res.get("budget", 8)
    depth = res.get("depth", 0.0)
    mass = res.get("mass", 0.0)
    hints = res.get("hints", 0)
    gates = res.get("gates", [])
    shadows = res.get("shadows", [])
    us = res.get("us", 0)
    prof = res.get("profile", "PERF")
    cpu_mhz = res.get("cpu_mhz", 240)
    saved_steps = budget - steps
    saved_pct = (saved_steps / budget) * 100.0 if budget > 0 else 0

    print("\n" + "=" * 70)
    print("  ESP32 AAGM-RLM ON-DEVICE INFERENCE ENGINE (HARDWARE-FREE SIM)")
    print("=" * 70)
    print(f"Input Text: \"{text}\"")
    print(f"Prediction: {sentiment}  |  Logits: [Neg: {logits[0]:.4f}, Pos: {logits[1]:.4f}]")
    print("-" * 70)
    print(f"Hardware Profile  : {prof} @ {cpu_mhz} MHz (Budget: {budget} steps max)")
    print(f"Latency           : {us} us ({us/1000.0:.2f} ms)")
    print(f"Recursion Steps   : {steps} / {budget}  ({saved_pct:.1f}% reduction vs fixed depth)")
    print(f"Soft Depth        : {depth:.4f}  |  Cumulative Halting Mass: {mass:.4f} / 0.900")
    print(f"Shadow Hints      : {hints} (prearm output path early)")
    print("-" * 70)
    print(f"{'Step':<6} {'Mixing Gate (g_k)':<20} {'Halting Mass':<18} {'Shadow (s_k)':<16} {'Action'}")
    print("-" * 70)

    cum_mass = 0.0
    for i in range(len(gates)):
        g = gates[i]
        s = shadows[i] if i < len(shadows) else 0.0
        cum_mass += (1.0 - g)
        hint_str = " [PREARM]" if s < 0.347 else ""
        if i + 1 == steps:
            action = f"HALT (Threshold >= 0.90){hint_str}"
        else:
            action = f"Continue{hint_str}"
        print(f"#{i+1:<5} {g:<20.4f} {cum_mass:<18.4f} {s:<16.4f} {action}")

    print("=" * 70)
    print("Memory & Storage Footprint:")
    print("  - Int8 Weights (Flash .rodata) : 510 KiB (Zero flash wasted for vocab)")
    print("  - Activation RAM (SRAM)        : 28 KiB (Static buffers, 0 dynamic heap)")
    print(f"  - Theoretical Energy Ratio     : {steps/8.0:.2f}x of standard fixed-depth model")
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="ESP32 AAGM-RLM Interactive Demo")
    ap.add_argument("--text", type=str, help="Single sentence to classify")
    ap.add_argument("--mode", type=str, default="PERF", choices=["PERF", "BAL", "ECO"],
                    help="Operating profile (PERF=240MHz/B8, BAL=160MHz/B6, ECO=80MHz/B4)")
    ap.add_argument("--budget", type=int, default=8, help="Max recursion budget (1..8)")
    args = ap.parse_args()

    if args.text:
        res = run_infer(args.text, args.budget, args.mode)
        display_result(args.text, res)
        return

    print("\nWelcome to the ESP32 AAGM-RLM Interactive Demonstration!")
    print("Type any sentence to run inference (or 'q' to quit, 'mode ECO' to change profile).\n")

    current_mode = args.mode
    current_budget = 8 if current_mode == "PERF" else (6 if current_mode == "BAL" else 4)

    presets = [
        "the movie was a brilliant masterpiece with stunning visuals and a powerful story.",
        "an absolute disaster, boring and painfully slow from start to finish.",
        "it starts slow but the brilliant ending makes it absolutely worth watching.",
        "the acting is good yet the plot feels dull, hollow and predictable."
    ]
    print("Preset examples:")
    for i, p in enumerate(presets, 1):
        print(f"  [{i}] {p}")
    print()

    while True:
        try:
            user_input = input(f"AAGM-RLM [{current_mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break
        if user_input.isdigit() and 1 <= int(user_input) <= len(presets):
            user_input = presets[int(user_input) - 1]
        elif user_input.upper().startswith("MODE "):
            parts = user_input.upper().split()
            if len(parts) > 1 and parts[1] in ("PERF", "BAL", "ECO"):
                current_mode = parts[1]
                current_budget = 8 if current_mode == "PERF" else (6 if current_mode == "BAL" else 4)
                print(f"Profile switched to {current_mode} (Budget: {current_budget} steps max)")
                continue

        res = run_infer(user_input, current_budget, current_mode)
        display_result(user_input, res)


if __name__ == "__main__":
    main()
