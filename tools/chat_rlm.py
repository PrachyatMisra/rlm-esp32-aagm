#!/usr/bin/env python3
"""Interactive offline Edge-RLM sentiment test CLI (Mac/Linux/Windows).

Runs the repository's native C++ inference harness without an ESP32 board.
The compact RLM is a binary sentiment classifier, not a text-generation model.

Features:
  - Review sentiment prediction and confidence
  - Recursive halting depth and cumulative-mass telemetry
  - Simulated operating-profile and battery controls

Usage:
  python tools/chat_rlm.py
  python tools/chat_rlm.py --prompt "the acting was phenomenal and deeply moving"
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


def ensure_harness():
    if not HARNESS.exists():
        subprocess.run(["make", "-C", str(HARNESS.parent), "host_harness"],
                       check=True, capture_output=True)


def infer(text: str, budget: int = 8, profile: str = "PERF", batt_mv: float = 4000.0) -> dict:
    ensure_harness()
    cmd = [str(HARNESS), "--infer", text, "--budget", str(budget), "--profile", profile]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    res = json.loads(proc.stdout.strip())

    # Battery droop computation (R_int = 0.200 Ohm)
    current_ma = 50.0 if profile == "PERF" else (36.0 if profile == "BAL" else 27.0)
    droop_mv = (current_ma / 1000.0) * 0.200 * 1000.0
    loaded_mv = batt_mv - droop_mv
    res["droop_mv"] = round(droop_mv, 1)
    res["loaded_mv"] = round(loaded_mv, 1)
    res["batt_ocv"] = batt_mv
    return res


def format_inference_result(user_msg: str, res: dict) -> str:
    pred = int(res.get("pred", 0))
    logits = res.get("logits", [0.0, 0.0])
    maximum = max(logits)
    probabilities = [pow(2.718281828, value - maximum) for value in logits]
    confidence = 100.0 * probabilities[pred] / sum(probabilities)
    steps = int(res.get("steps", 0))
    mass = float(res.get("mass", 0.0))
    verdict = "POSITIVE" if pred == 1 else "NEGATIVE"
    saved = max(0, 8 - steps)
    return (
        f"Input: \"{user_msg}\"\n"
        f"Prediction: {verdict} ({confidence:.1f}% confidence)\n"
        f"Adaptive depth: {steps} steps; halting mass {mass:.3f} / 0.900\n"
        f"Host inference: {int(res.get('us', 0)):,} μs at {res.get('cpu_mhz')} MHz ({res.get('profile')})\n"
        f"Battery estimate: {res.get('loaded_mv')} mV loaded, {res.get('droop_mv')} mV droop\n"
        f"Compared with the 8-step budget: {saved} steps skipped. This classifier predicts review sentiment; it does not generate text."
    )

def main():
    ap = argparse.ArgumentParser(description="Offline Edge-RLM Sentiment Test")
    ap.add_argument("--prompt", type=str, help="Single prompt to evaluate")
    ap.add_argument("--mode", type=str, default="PERF", choices=["PERF", "BAL", "ECO"],
                    help="Operating profile (PERF=240M/8, BAL=160M/6, ECO=80M/4)")
    ap.add_argument("--batt", type=float, default=4000.0, help="Battery open-circuit voltage in mV")
    args = ap.parse_args()

    ensure_harness()

    if args.prompt:
        r = infer(args.prompt, 8 if args.mode == "PERF" else (6 if args.mode == "BAL" else 4),
                  args.mode, args.batt)
        print("\n" + format_inference_result(args.prompt, r) + "\n")
        return

    print("=" * 70)
    print("   Edge-RLM Offline Sentiment Test (ESP32 Host Simulation)")
    print("=" * 70)
    print("Commands:")
    print("  /mode PERF|BAL|ECO  - Change DVFS profile")
    print("  /batt <millivolts>  - Adjust simulated battery voltage (e.g. /batt 3500)")
    print("  /help               - Display command reference")
    print("  exit | quit         - Exit the test shell")
    print("-" * 70)
    print("Paste a review or sentence to run binary sentiment classification:\n")

    current_mode = args.mode
    current_batt = args.batt
    budget = 8 if current_mode == "PERF" else (6 if current_mode == "BAL" else 4)

    while True:
        try:
            line = input(f"RLM [{current_mode} | {current_batt:.0f}mV] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting sentiment test.")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) > 1 and parts[1].upper() in ("PERF", "BAL", "ECO"):
                current_mode = parts[1].upper()
                budget = 8 if current_mode == "PERF" else (6 if current_mode == "BAL" else 4)
                print(f"[OK] Mode switched to {current_mode} (Max Budget: {budget} steps)")
            else:
                print("[!] Usage: /mode PERF | BAL | ECO")
            continue
        if line.startswith("/batt"):
            parts = line.split()
            if len(parts) > 1:
                try:
                    current_batt = float(parts[1])
                    print(f"[OK] Battery set to {current_batt:.0f} mV")
                except ValueError:
                    print("[!] Invalid voltage.")
            continue
        if line == "/help":
            print("Available commands: /mode <PERF|BAL|ECO>, /batt <mV>, exit")
            continue

        res = infer(line, budget, current_mode, current_batt)
        reply = format_inference_result(line, res)
        print("\n" + reply + "\n")


if __name__ == "__main__":
    main()
