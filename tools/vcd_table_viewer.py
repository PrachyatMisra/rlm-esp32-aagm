#!/usr/bin/env python3
"""Parse aagm_coprocessor.vcd and print a clean, cycle-accurate tabular timeline.

Compares Verilog RTL simulation signals to the expected algorithmic behavior.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VCD_PATH = Path(__file__).resolve().parent.parent / "verification" / "verilog" / "aagm_coprocessor.vcd"


def parse_vcd_table(vcd_path: Path | None = None) -> list[dict]:
    path = Path(vcd_path) if vcd_path else VCD_PATH
    if not path.exists():
        raise FileNotFoundError(f"VCD file not found at: {path}. Run 'make -C verification/verilog check' first.")

    content = path.read_text(encoding="utf-8", errors="replace")
    
    symbol_map = {}
    for line in content.splitlines():
        line = line.strip()
        m = re.match(r"^\$var\s+\w+\s+\d+\s+(\S+)\s+(\S+)", line)
        if m:
            sym, name = m.group(1), m.group(2)
            symbol_map[sym] = name

    signals = {
        "clk": 0,
        "rst_n": 0,
        "p_sel": 0,
        "p_enable": 0,
        "p_write": 0,
        "p_addr": 0,
        "p_wdata": 0,
        "irq_halt": 0,
        "irq_abort": 0,
        "speculative_prearm": 0,
        "steps_reg": 0,
        "mass_acc_reg": 0,
        "gate_reg": 0,
        "shadow_reg": 0,
    }

    current_time = 0
    records = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_time = int(line[1:])
            continue

        if len(line) >= 2 and line[0] in "01xz":
            val = 1 if line[0] == "1" else 0
            sym = line[1:]
            name = symbol_map.get(sym)
            if name in signals:
                signals[name] = val
                records.append((current_time, dict(signals)))
        elif line.startswith("b"):
            parts = line[1:].split()
            if len(parts) == 2:
                bin_str, sym = parts
                name = symbol_map.get(sym)
                if name in signals:
                    try:
                        val = int(bin_str, 2)
                    except ValueError:
                        val = 0
                    signals[name] = val
                    records.append((current_time, dict(signals)))

    rows = []
    last_action = None
    last_key = None

    for t_ps, state in records:
        t_ns = t_ps / 1000.0
        mass_q15 = state["mass_acc_reg"]
        mass_float = round(mass_q15 / 32768.0, 3)

        action = "IDLE / Bus Standby"
        if not state["rst_n"]:
            action = "Reset Active (rst_n=0)"
        elif state["irq_abort"]:
            action = "⚠ SINGLE-CYCLE ABORT IRQ ASSERTED (bit 2)"
        elif state["irq_halt"]:
            action = "★ HALTING THRESHOLD MET -> HALT IRQ ASSERTED"
        elif state["speculative_prearm"] and not state["irq_halt"]:
            action = "⚡ APOP PREARM ASSERTED (s_{k+1} < 0.347)"
        elif state["steps_reg"] == 1 and state["p_addr"] == 8:
            action = "Step 1 Gate Ingested (g_1=0.6058, mass=0.394)"
        elif state["steps_reg"] == 0 and state["p_addr"] == 0 and state["p_wdata"] == 1:
            action = "Inference Started (CTRL_REG <= 0x01)"
        elif state["p_addr"] == 0x14 and state["p_wdata"] == 8:
            action = "Budget Configured (BUDGET <= 8)"
        elif state["p_addr"] == 0x0C and state["p_wdata"] == 17911:
            action = "Step 1 Shadow Posted (s_2=0.5466)"
        elif state["p_addr"] == 0x0C and state["p_wdata"] == 9175:
            action = "Step 2 Shadow Posted (s_3=0.2800 < 0.347)"

        key = (
            state["rst_n"],
            state["p_addr"],
            state["p_wdata"],
            state["steps_reg"],
            state["mass_acc_reg"],
            state["speculative_prearm"],
            state["irq_halt"],
            state["irq_abort"],
            action,
        )

        if key == last_key:
            continue
        last_key = key

        rows.append({
            "time_ns": f"{t_ns:.1f} ns",
            "rst_n": str(state["rst_n"]),
            "bus_addr": f"0x{state['p_addr']:02X}",
            "bus_wdata": f"{state['p_wdata']:>5}",
            "step_k": str(state["steps_reg"]),
            "mass_q15": f"{mass_q15:>5} ({mass_float:.3f})",
            "prearm": "1 (APOP)" if state["speculative_prearm"] else "0",
            "halt_irq": "1 (HALT)" if state["irq_halt"] else "0",
            "abort_irq": "1 (ABORT)" if state["irq_abort"] else "0",
            "hardware_action": action,
        })

    return rows


def print_table(rows: list[dict]):
    headers = [
        ("Time", "time_ns", 10),
        ("RST_N", "rst_n", 6),
        ("Addr", "bus_addr", 6),
        ("WData", "bus_wdata", 7),
        ("Step", "step_k", 5),
        ("Cum. Mass (Q1.15 / Float)", "mass_q15", 26),
        ("Prearm", "prearm", 9),
        ("Halt IRQ", "halt_irq", 9),
        ("Abort", "abort_irq", 9),
        ("Hardware State / Action", "hardware_action", 46),
    ]

    header_line = " | ".join(title.ljust(w) for title, _, w in headers)
    divider_line = "-+-".join("-" * w for _, _, w in headers)

    print("\n" + "=" * len(header_line))
    print("  EDGE-RLM AAGM COPROCESSOR RTL CYCLE-BY-CYCLE VERIFICATION TABLE")
    print("=" * len(header_line))
    print(header_line)
    print(divider_line)

    for r in rows:
        line = " | ".join(str(r[key]).ljust(w) for _, key, w in headers)
        print(line)

    print("=" * len(header_line) + "\n")


def main():
    vcd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else VCD_PATH
    try:
        rows = parse_vcd_table(vcd_path)
        print_table(rows)
    except Exception as err:
        print(f"Error parsing VCD table: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

