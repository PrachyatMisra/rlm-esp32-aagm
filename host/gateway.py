"""Host gateway + analysis suite for the ESP32 AAGM-RLM.

Usage:
    python gateway.py --port /dev/ttyUSB0              # interactive bench
    python gateway.py --port /dev/ttyUSB0 --eval       # labeled demo set
    python gateway.py --selftest                       # native firmware parity
                                                       (builds & runs the
                                                       host_harness, no board
                                                       needed)

Artifacts written to <repo>/artifacts/:
    esp32_parity.json            native-firmware vs PyTorch mirror parity
    esp32_eval_telemetry.json    device inference records (real hardware)
    esp32_gating_analysis.png    4-panel mechanism dashboard
    esp32_energy_model.png       mA / mJ estimates per mode (model-based)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESP32_ROOT = HERE.parent
REPO_ROOT = ESP32_ROOT.parent
ART = REPO_ROOT / "artifacts"

DEMO_SET = [
    ("the movie was a brilliant masterpiece with stunning visuals and a powerful story.", 1),
    ("an absolute disaster, boring and painfully slow from start to finish.", 0),
    ("it starts slow but the brilliant ending makes it absolutely worth watching.", 1),
    ("great cast but the terrible script ruins the whole film completely.", 0),
    ("this film is not bad at all, i actually enjoyed every minute of it.", 1),
    ("the acting is good yet the plot feels dull, hollow and predictable.", 0),
    ("short review: superb.", 1),
    ("bad.", 0),
]


# ----------------------------------------------------------------------
def selftest() -> dict:
    """Build the real firmware engine natively and run golden parity."""
    tw = ESP32_ROOT / "firmware" / "test"
    subprocess.run(["make", "-C", str(tw), "host_harness"],
                   check=True, capture_output=True)
    gold = tw.parent.parent / "tools" / "out" / "golden_vectors.txt"
    out = subprocess.run([str(tw / "host_harness"), str(gold), "--bench", "30"],
                         check=True, capture_output=True, text=True).stdout
    report = json.loads(out)
    ART.mkdir(exist_ok=True)
    (ART / "esp32_parity.json").write_text(json.dumps(report, indent=2))
    s = report["summary"]
    print(f"[selftest] {s['passed']}/{s['golden']} golden vectors passed "
          f"(max logit err {s['max_logit_err']}, PASS={s['PASS']})")
    return report


class Device:
    """Line-based serial client for the firmware protocol."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 5.0):
        import serial
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(2.0)                       # ESP32 resets on port open
        self.ser.reset_input_buffer()
        self.banner = self._read_line(5.0)

    def _read_line(self, timeout: float) -> str:
        end = time.time() + timeout
        while time.time() < end:
            line = self.ser.readline().decode("utf-8", "replace").strip()
            if line:
                return line
        return ""

    def cmd(self, line: str, timeout: float = 120.0) -> dict:
        self.ser.write((line + "\n").encode())
        raw = self._read_line(timeout)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw, "err": "parse"}

    def infer(self, text: str) -> dict:
        return self.cmd(f"INFER {text}")


# ----------------------------------------------------------------------
def eval_on_device(dev: Device) -> dict:
    tele = {
        "banner": dev.banner,
        "stat": dev.cmd("STAT"),
        "records": [],
        "mode_sweeps": {},
    }
    for text, expect in DEMO_SET:
        r = dev.infer(text)
        r["expected"] = expect
        tele["records"].append(r)
        print(f"  pred={r.get('pred')} exp={expect} steps={r.get('steps')} "
              f"us={r.get('us')} | {text[:50]}")
    for mode in ("PERF", "BAL", "ECO"):
        dev.cmd(f"MODE {mode}")
        tele["mode_sweeps"][mode] = dev.cmd("BENCH 5", timeout=600)
        print(f"  bench {mode}: {tele['mode_sweeps'][mode]}")
    dev.cmd("MODE AUTO")
    ART.mkdir(exist_ok=True)
    (ART / "esp32_eval_telemetry.json").write_text(json.dumps(tele, indent=2))
    return tele


# ----------------------------------------------------------------------
def make_figures(soft: dict, parity: dict | None, tele: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("AAGM-RLM on ESP32 - mechanism dashboard", fontsize=14)

    ax = axes[0][0]
    steps_a = soft["test_aagm"]["mean_steps"]
    steps_f = soft["test_fixed_depth8"]["mean_steps"]
    accs = [soft["test_aagm"]["accuracy"], soft["test_fixed_depth8"]["accuracy"],
            soft["test_int8_mirror_aagm"]["accuracy"]]
    ax.bar(["AAGM+ACT\n(adaptive halting)", "Fixed depth 8", "int8 mirror\n(= firmware)"],
           [steps_a, steps_f, soft["test_int8_mirror_aagm"]["mean_steps"]],
           color=["#1f77b4", "#d62728", "#2ca02c"])
    ax.set_ylabel("mean recursion steps / inference (of 8 max)")
    for i, a in enumerate(accs):
        ax.annotate(f"acc {a:.2%}", (i, 0.2), ha="center", color="white",
                    fontsize=10, fontweight="bold")
    saved = (1 - steps_a / steps_f) * 100
    ax.set_title(f"Adaptive halting: {saved:.0f}% fewer steps at equal accuracy")

    ax = axes[0][1]
    budgets = sorted(int(b) for b in soft["test_by_budget"])
    ax.plot(budgets, [soft["test_by_budget"][str(b)]["accuracy"] * 100 for b in budgets],
            "o-", label="accuracy %", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(budgets, [budgets[0] / b * 100 for b in budgets], "s--",
             label="energy share % (1/budget)", color="#d62728")
    ax.set_xlabel("energy-context recursion budget (PERF=8 BAL=6 ECO=4)")
    ax.set_ylabel("accuracy %")
    ax2.set_ylabel("relative step energy %", color="#d62728")
    ax.set_title("Hardware-level adaptive gating: budget vs accuracy")

    ax = axes[1][0]
    sweep = soft["calibration"]["final"]["sweep"]
    taus = [r["tau_lo"] for r in sweep]
    ax.plot(taus, [r["precision"] for r in sweep], label="hint precision")
    ax.plot(taus, [r["hint_rate"] for r in sweep], label="hint rate")
    ax.axvline(soft["config"]["tau_lo"], ls="--", color="gray",
               label=f"chosen $\\tau_{{lo}}$={soft['config']['tau_lo']}")
    ax.set_xlabel(r"shadow retire threshold $\tau_{lo}$")
    ax.set_title("Shadow-gate calibration (retire-hint scheduler)")
    ax.legend()
    ax.set_xlim(0, max(taus))

    ax = axes[1][1]
    if parity is not None:
        s = parity["summary"]
        labels = ["max logit err", "max gate err", "max shadow err"]
        vals = [s["max_logit_err"], s["max_gate_err"], s["max_shadow_err"]]
        ax.bar(labels, [max(v, 1e-9) for v in vals], color="#2ca02c")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1e-2)
        ax.set_title(f"Host-native firmware parity: {s['passed']}/{s['golden']} "
                     f"vectors exact decisions")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1e}".replace("e-0", "e-"), (i, max(v, 1e-9)),
                        ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = ART / "esp32_gating_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # model-based energy panel (annotated estimates from cited measurements)
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = ["PERF 240MHz B8", "BAL 160MHz B6", "ECO 80MHz B4"]
    ma = [50, 36, 27]             # dual-core active draw per Espressif data
    scale = [8 / steps_a, 6 / steps_a, 4 / steps_a]   # time at cap vs adaptive
    mj = [m * s for m, s in zip(ma, scale)]
    x = range(3)
    ax.bar([i - 0.2 for i in x], ma, 0.4,
           label="CPU current draw mA (typ.)", color="#1f77b4")
    ax.bar([i + 0.2 for i in x], mj, 0.4,
           label="energy index: mA x budget/mean_steps (relative mJ)", color="#ff7f0e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_title("Model-based energy estimate per DMMS profile "
                 "(calibrate with INA219 on real bench)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "esp32_energy_model.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port of the ESP32, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--eval", action="store_true",
                    help="run labeled demo set + mode sweep on the device")
    ap.add_argument("--selftest", action="store_true",
                    help="native firmware parity (no hardware needed)")
    args = ap.parse_args()

    soft_path = ART / "esp32_software_results.json"
    soft = json.loads(soft_path.read_text()) if soft_path.exists() else None

    parity = None
    if args.selftest or not args.port:
        parity = selftest()

    tele = None
    if args.port:
        dev = Device(args.port, args.baud)
        print("[dev]", dev.banner)
        if args.eval:
            tele = eval_on_device(dev)
        else:
            for text, expect in DEMO_SET[:4]:
                r = dev.infer(text)
                print(json.dumps(r))

    if soft is not None and (parity or tele):
        make_figures(soft, parity, tele)
        print("[done] figures -> artifacts/esp32_gating_analysis.png, "
              "esp32_energy_model.png")


if __name__ == "__main__":
    main()
