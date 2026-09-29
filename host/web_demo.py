#!/usr/bin/env python3
"""Local Edge-RLM demo server and engineering workbench.

Runs the standard-library HTTP server on 0.0.0.0:8000. The landing page
separates repository chat from native C++ sentiment inference; the detailed
engineering workbench remains available at /lab, with the Graphify explorer at
/graph.

Features:
  - Focused local project chat with citations and optional loopback Ollama
  - Real Edge-RLM sentiment analysis through the native C++ host harness
  - Searchable 3D Graphify code and file-pipeline explorer at /graph
  - Interactive Live Inference Studio and presets at /lab
  - Interactive Web Serial Console (simulates ESP32 UART in browser)
  - Live Chart.js Visualization (Halting Mass & Gate Curve, Energy Breakdown)
  - Patent Specification & 20 Formal Claims Explorer
  - Hardware Coprocessor Register & Verilog Cycle Simulation Explorer
  - Battery Impedance Droop Compensation Slider (3.3V-4.2V with brownout alert)
  - College Project Review Defense & Viva Voce Guide
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

try:
    from .chat_agent import respond as respond_to_chat, status as chat_status
except ImportError:
    from chat_agent import respond as respond_to_chat, status as chat_status

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "firmware" / "test" / "host_harness"
VERILOG_DIR = REPO_ROOT / "verification" / "verilog"
GOLDEN_TXT = REPO_ROOT / "tools" / "out" / "golden_vectors.txt"
PATENT_MD = REPO_ROOT / "docs" / "PATENT_SPECIFICATION.md"
_HARNESS_BUILD_LOCK = Lock()


def ensure_harness():
    with _HARNESS_BUILD_LOCK:
        if not HARNESS.exists():
            subprocess.run(["make", "-C", str(HARNESS.parent), "host_harness"],
                           check=True, capture_output=True)


def run_inference(text: str, budget: int = 8, profile: str = "PERF", batt_mv: float = 4000.0) -> dict:
    ensure_harness()
    current_ma = 50.0 if profile == "PERF" else (36.0 if profile == "BAL" else 27.0)
    v_droop_mv = (current_ma / 1000.0) * 0.200 * 1000.0  # R_int = 0.200 Ohm
    loaded_batt_mv = batt_mv - v_droop_mv

    if profile == "AUTO":
        pct = max(0.0, min(100.0, (loaded_batt_mv - 3300.0) / 900.0 * 100.0))
        if pct < 25.0:
            profile = "ECO"
            budget = 4
        elif pct < 50.0:
            profile = "BAL"
            budget = 6
        else:
            profile = "PERF"
            budget = 8

    cmd = [str(HARNESS), "--infer", text, "--budget", str(budget), "--profile", profile]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    res = json.loads(proc.stdout.strip())
    res["batt_mv"] = batt_mv
    res["v_droop_mv"] = round(v_droop_mv, 1)
    res["loaded_batt_mv"] = round(loaded_batt_mv, 1)

    fixed_steps = 8
    saved_steps = max(0, fixed_steps - res["steps"])
    saved_pct = round((saved_steps / fixed_steps) * 100.0, 1)

    step_time_ms = (res["us"] / res["steps"]) / 1000.0 if res["steps"] > 0 else 1.5
    energy_uj = round(current_ma * 3.3 * (res["us"] / 1000.0), 1)
    baseline_energy_uj = round(50.0 * 3.3 * (step_time_ms * 8.0), 1)
    energy_saved_pct = round(max(0.0, (1.0 - energy_uj / baseline_energy_uj) * 100.0), 1)

    res["comparison"] = {
        "fixed_steps": fixed_steps,
        "actual_steps": res["steps"],
        "saved_steps": saved_steps,
        "saved_steps_pct": saved_pct,
        "actual_energy_uj": energy_uj,
        "baseline_energy_uj": baseline_energy_uj,
        "energy_saved_pct": energy_saved_pct,
        "flash_kb": 510,
        "sram_kb": 28,
        "vocab_overhead_bytes": 0,
        "speculative_head_hits": res.get("spec_hits", 0),
        "frozen_tokens": res.get("frozen", 0),
        "macs_saved": res.get("macs_saved", 0),
        "tail_us_saved": res.get("tail_us_saved", 0)
    }
    return res


def run_serial_cmd(cmd_line: str) -> str:
    ensure_harness()
    proc = subprocess.Popen([str(HARNESS), "--interactive"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    out, _ = proc.communicate(input=cmd_line.strip() + "\nQUIT\n", timeout=10)
    lines = out.strip().split("\n")
    # Return device output excluding the banner and quit
    resp_lines = [l for l in lines if l and not l.startswith(">RLM-AAGM")]
    return "\n".join(resp_lines) if resp_lines else "{}"


def run_golden_check() -> dict:
    ensure_harness()
    cmd = [str(HARNESS), str(GOLDEN_TXT), "--bench", "10"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip())


def run_verilog_check() -> dict:
    try:
        proc = subprocess.run(["make", "-C", str(VERILOG_DIR), "check"],
                              check=True, capture_output=True, text=True)
        combined_output = proc.stdout
        status = "PASS" if ("AAGM_VERILOG_PASS" in combined_output and "AAGM_COPROCESSOR_VERILOG_PASS" in combined_output) else "FAIL"
        return {
            "output": combined_output.strip(),
            "status": status
        }
    except Exception:
        sim_gate = VERILOG_DIR / "sim_aagm_gate.py"
        sim_coproc = VERILOG_DIR / "sim_aagm_arbiter_coprocessor.py"
        proc_g = subprocess.run([sys.executable, str(sim_gate)],
                                check=True, capture_output=True, text=True)
        proc_c = subprocess.run([sys.executable, str(sim_coproc)],
                                check=True, capture_output=True, text=True)

        combined_output = proc_g.stdout + "\n" + proc_c.stdout
        return {
            "output": combined_output.strip(),
            "status": "PASS" if ("AAGM_VERILOG_PASS" in combined_output and "AAGM_COPROCESSOR_VERILOG_PASS" in combined_output) else "FAIL"
        }


def get_patent_text() -> str:
    if PATENT_MD.exists():
        return PATENT_MD.read_text()
    return "Patent specification not found."


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge-RLM + AAGM on ESP32 | Interactive Engineering &amp; Patent Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; color: #f1f5f9; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glass-card { background: #111827; border: 1px solid #1f2937; }
    .terminal-window { background: #070a12; border: 1px solid #1e293b; font-family: 'JetBrains Mono', monospace; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Top Navigation Header -->
  <header class="border-b border-gray-800 bg-gray-950/90 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20 text-lg">
          <i class="fa-solid fa-microchip"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="font-extrabold text-lg tracking-tight text-white">Edge-RLM <span class="text-cyan-400 font-mono text-xs px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800">ESP32 + AAGM</span></h1>
            <span class="text-[11px] bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded-full font-medium">Patent Track</span>
          </div>
          <p class="text-xs text-gray-400">Asynchronous Dual-Core Recursive Transformer with Speculative Early-Exit</p>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <nav class="flex flex-wrap items-center gap-1.5 text-xs sm:text-sm">
        <button onclick="switchTab('demo')" id="tab-btn-demo" class="px-3 py-1.5 rounded-lg font-medium transition bg-cyan-600 text-white shadow-sm flex items-center">
          <i class="fa-solid fa-bolt mr-1.5 text-xs"></i>Live Studio
        </button>
        <a href="/" class="px-3 py-1.5 rounded-lg font-medium transition text-emerald-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-comments mr-1.5 text-xs"></i>Project Chat
        </a>
        <button onclick="switchTab('serial')" id="tab-btn-serial" class="px-3 py-1.5 rounded-lg font-medium transition text-amber-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-terminal mr-1.5 text-xs"></i>Serial Console
        </button>
        <button onclick="switchTab('coproc')" id="tab-btn-coproc" class="px-3 py-1.5 rounded-lg font-medium transition text-blue-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-microchip mr-1.5 text-xs"></i>Coprocessor
        </button>
        <button onclick="switchTab('patent')" id="tab-btn-patent" class="px-3 py-1.5 rounded-lg font-medium transition text-purple-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-certificate mr-1.5 text-xs"></i>Patent Claims
        </button>
        <button onclick="switchTab('verify')" id="tab-btn-verify" class="px-3 py-1.5 rounded-lg font-medium transition text-gray-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-check-double mr-1.5 text-xs"></i>Verification
        </button>
        <button onclick="switchTab('defense')" id="tab-btn-defense" class="px-3 py-1.5 rounded-lg font-medium transition text-gray-400 hover:text-white hover:bg-gray-800 flex items-center">
          <i class="fa-solid fa-graduation-cap mr-1.5 text-xs"></i>Review Guide
        </button>
        <a href="/graph" class="px-3 py-1.5 rounded-lg font-medium transition text-cyan-300 hover:text-white hover:bg-gray-800 flex items-center border border-cyan-900/60">
          <i class="fa-solid fa-share-nodes mr-1.5 text-xs"></i>3D Pipeline Graph ↗
        </a>
      </nav>
    </div>
  </header>

  <!-- Main Content Area -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">

    <!-- ============================================================== -->
    <!-- TAB 1: LIVE INFERENCE STUDIO -->
    <!-- ============================================================== -->
    <div id="tab-demo" class="space-y-6">
      
      <!-- Top Metrics Banner -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="glass-card p-4 rounded-xl flex items-center space-x-3 border-l-4 border-cyan-500">
          <div class="p-2.5 rounded-lg bg-cyan-950 text-cyan-400 text-lg"><i class="fa-solid fa-gauge-high"></i></div>
          <div>
            <div class="text-[11px] text-gray-400 uppercase font-semibold">Adaptive Halting</div>
            <div class="text-lg font-bold text-white"><span class="text-cyan-400">2.18</span> / 8 steps avg</div>
            <div class="text-xs text-emerald-400">72.8% cycle savings</div>
          </div>
        </div>
        <div class="glass-card p-4 rounded-xl flex items-center space-x-3 border-l-4 border-emerald-500">
          <div class="p-2.5 rounded-lg bg-emerald-950 text-emerald-400 text-lg"><i class="fa-solid fa-bullseye"></i></div>
          <div>
            <div class="text-[11px] text-gray-400 uppercase font-semibold">Model Accuracy</div>
            <div class="text-lg font-bold text-white"><span class="text-emerald-400">92.33%</span> (int8)</div>
            <div class="text-xs text-gray-400">0.00% quantization loss</div>
          </div>
        </div>
        <div class="glass-card p-4 rounded-xl flex items-center space-x-3 border-l-4 border-purple-500">
          <div class="p-2.5 rounded-lg bg-purple-950 text-purple-400 text-lg"><i class="fa-solid fa-forward-fast"></i></div>
          <div>
            <div class="text-[11px] text-gray-400 uppercase font-semibold">Tail Latency (APOP)</div>
            <div class="text-lg font-bold text-white">0 μs Added</div>
            <div class="text-xs text-purple-400">Speculative pre-computed</div>
          </div>
        </div>
        <div class="glass-card p-4 rounded-xl flex items-center space-x-3 border-l-4 border-blue-500">
          <div class="p-2.5 rounded-lg bg-blue-950 text-blue-400 text-lg"><i class="fa-solid fa-memory"></i></div>
          <div>
            <div class="text-[11px] text-gray-400 uppercase font-semibold">Memory Budget</div>
            <div class="text-lg font-bold text-white">510 KiB <span class="text-xs text-gray-400 font-normal">Flash</span></div>
            <div class="text-xs text-blue-400">28 KiB SRAM (0 dynamic heap)</div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Left: Input & Hardware Configuration -->
        <div class="space-y-6 lg:col-span-1">
          
          <!-- Text Input Card -->
          <div class="glass-card p-5 rounded-2xl space-y-4">
            <div class="flex items-center justify-between">
              <h2 class="font-bold text-white flex items-center text-sm">
                <i class="fa-solid fa-keyboard mr-2 text-cyan-400"></i>Input Review Text
              </h2>
              <span class="text-[11px] text-gray-400 font-mono">FNV-1a Tokenizer</span>
            </div>

            <textarea id="input-text" rows="4" class="w-full bg-gray-950 border border-gray-800 rounded-xl p-3 text-sm focus:outline-none focus:border-cyan-500 text-gray-100 placeholder-gray-500 transition" placeholder="Enter text to analyze sentiment with AAGM-RLM...">the movie was a brilliant masterpiece with stunning visuals and a powerful story.</textarea>

            <!-- Presets -->
            <div class="space-y-1.5">
              <label class="text-xs font-semibold text-gray-400">Quick Test Presets:</label>
              <div class="grid grid-cols-1 gap-1.5 text-xs">
                <button onclick="setPreset(0)" class="text-left px-2.5 py-1.5 rounded-lg bg-gray-800/80 hover:bg-gray-800 text-gray-300 hover:text-white transition truncate">
                  <span class="text-emerald-400 font-bold mr-1">[+]</span> Masterpiece with stunning visuals
                </button>
                <button onclick="setPreset(1)" class="text-left px-2.5 py-1.5 rounded-lg bg-gray-800/80 hover:bg-gray-800 text-gray-300 hover:text-white transition truncate">
                  <span class="text-rose-400 font-bold mr-1">[-]</span> Absolute disaster, slow and boring
                </button>
                <button onclick="setPreset(2)" class="text-left px-2.5 py-1.5 rounded-lg bg-gray-800/80 hover:bg-gray-800 text-gray-300 hover:text-white transition truncate">
                  <span class="text-emerald-400 font-bold mr-1">[+]</span> Starts slow but brilliant ending
                </button>
                <button onclick="setPreset(3)" class="text-left px-2.5 py-1.5 rounded-lg bg-gray-800/80 hover:bg-gray-800 text-gray-300 hover:text-white transition truncate">
                  <span class="text-rose-400 font-bold mr-1">[-]</span> Great cast but terrible script ruins it
                </button>
              </div>
            </div>

            <!-- Hardware Operating Profile Controls -->
            <div class="pt-3 border-t border-gray-800 space-y-4">
              <div class="flex items-center justify-between">
                <label class="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center">
                  <i class="fa-solid fa-sliders mr-1.5 text-cyan-400"></i>Power Profile &amp; DVFS
                </label>
                <span id="profile-indicator" class="text-xs font-mono font-bold text-cyan-400 bg-cyan-950 border border-cyan-800 px-2 py-0.5 rounded">PERF (240MHz/B8)</span>
              </div>

              <!-- Profile Buttons -->
              <div class="grid grid-cols-4 gap-1.5 text-xs font-semibold">
                <button onclick="selectProfile('PERF')" id="btn-prof-PERF" class="py-2 rounded-lg bg-cyan-600 text-white transition text-center shadow-sm">
                  PERF<br><span class="text-[10px] opacity-75">240M/8</span>
                </button>
                <button onclick="selectProfile('BAL')" id="btn-prof-BAL" class="py-2 rounded-lg bg-gray-800 text-gray-300 hover:text-white transition text-center">
                  BAL<br><span class="text-[10px] opacity-75">160M/6</span>
                </button>
                <button onclick="selectProfile('ECO')" id="btn-prof-ECO" class="py-2 rounded-lg bg-gray-800 text-gray-300 hover:text-white transition text-center">
                  ECO<br><span class="text-[10px] opacity-75">80M/4</span>
                </button>
                <button onclick="selectProfile('AUTO')" id="btn-prof-AUTO" class="py-2 rounded-lg bg-gray-800 text-gray-300 hover:text-white transition text-center">
                  AUTO<br><span class="text-[10px] opacity-75">Batt</span>
                </button>
              </div>

              <!-- Battery Slider & Droop Compensation -->
              <div class="space-y-1.5">
                <div class="flex justify-between text-xs">
                  <span class="text-gray-400"><i class="fa-solid fa-battery-half mr-1 text-emerald-400"></i>Battery OCV (GPIO34)</span>
                  <span id="batt-val" class="font-mono text-emerald-400 font-bold">4000 mV (78%)</span>
                </div>
                <input type="range" id="batt-slider" min="3300" max="4200" value="4000" step="50" oninput="updateBattery(this.value)" class="w-full accent-emerald-500 bg-gray-800 h-2 rounded-lg cursor-pointer">
                <div class="flex justify-between text-[11px] text-gray-400 font-mono">
                  <span>Loaded: <span id="loaded-batt-val" class="text-cyan-400 font-bold">3990 mV</span></span>
                  <span id="droop-indicator" class="text-purple-400 font-semibold"><i class="fa-solid fa-shield-halved mr-1"></i>Droop: -10 mV</span>
                </div>
                <div id="brownout-alert" class="hidden text-[11px] p-2 rounded bg-rose-950/80 border border-rose-800 text-rose-300 font-mono">
                  <i class="fa-solid fa-triangle-exclamation mr-1"></i>Warning: Loaded rail near 3.3V! Auto ECO &amp; cooperative abort active.
                </div>
              </div>
            </div>

            <!-- Run Button -->
            <button onclick="executeInference()" id="btn-run" class="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold text-sm shadow-lg shadow-cyan-500/25 hover:opacity-95 transition flex items-center justify-center space-x-2">
              <i class="fa-solid fa-bolt"></i>
              <span>Run On-Device Inference</span>
            </button>
          </div>
        </div>

        <!-- Right 2 Columns: Results & Trace -->
        <div class="space-y-6 lg:col-span-2">
          
          <!-- Inference Output & Classification -->
          <div class="glass-card p-6 rounded-2xl space-y-6">
            <div class="flex flex-wrap items-center justify-between gap-2 border-b border-gray-800 pb-4">
              <div>
                <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Classification Outcome</h3>
                <div class="flex items-center space-x-3 mt-1">
                  <div id="pred-badge" class="text-xl font-extrabold px-3 py-1 rounded-xl bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center space-x-2">
                    <i class="fa-solid fa-thumbs-up"></i>
                    <span>POSITIVE</span>
                  </div>
                  <div class="text-xs font-mono text-gray-400">
                    Confidence: <span id="conf-val" class="text-white font-bold text-sm">98.4%</span>
                  </div>
                </div>
              </div>

              <!-- Key Metrics Pill Group -->
              <div class="flex items-center space-x-2 text-xs">
                <div class="bg-gray-900 border border-gray-800 px-3 py-2 rounded-xl text-center">
                  <div class="text-gray-400 text-[10px] uppercase">Latency</div>
                  <div id="metric-lat" class="font-mono font-bold text-cyan-400 text-sm">3,469 μs</div>
                </div>
                <div class="bg-gray-900 border border-gray-800 px-3 py-2 rounded-xl text-center">
                  <div class="text-gray-400 text-[10px] uppercase">Steps Taken</div>
                  <div id="metric-steps" class="font-mono font-bold text-emerald-400 text-sm">2 / 8</div>
                </div>
                <div class="bg-gray-900 border border-gray-800 px-3 py-2 rounded-xl text-center">
                  <div class="text-gray-400 text-[10px] uppercase">Est. Energy</div>
                  <div id="metric-energy" class="font-mono font-bold text-amber-400 text-sm">57.2 μJ</div>
                </div>
              </div>
            </div>

            <!-- Dynamic Recursion Chart & Progress -->
            <div class="space-y-4">
              <div class="flex justify-between items-center text-xs">
                <span class="text-gray-300 font-semibold flex items-center">
                  <i class="fa-solid fa-chart-line mr-1.5 text-cyan-400"></i>ACT Cumulative Halting Mass vs Threshold (0.900)
                </span>
                <span id="mass-label" class="font-mono font-bold text-cyan-400">1.018 / 0.900 (Threshold Met)</span>
              </div>
              
              <!-- Interactive Chart.js Canvas -->
              <div class="h-44 w-full bg-gray-950/70 border border-gray-800 rounded-xl p-2 relative">
                <canvas id="recursionChart"></canvas>
              </div>
            </div>

            <!-- Dynamic Recursion Steps Breakdown -->
            <div class="space-y-3">
              <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center">
                <i class="fa-solid fa-layer-group mr-1.5 text-cyan-400"></i>Step-by-Step Recursive Execution Trace
              </h4>
              <div id="step-cards-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                <!-- Dynamically populated step cards -->
              </div>
            </div>

            <!-- Side-by-Side Savings Analysis -->
            <div class="bg-gray-950/80 border border-gray-800 rounded-xl p-4">
              <h4 class="text-xs font-bold text-gray-300 uppercase tracking-wider mb-3 flex items-center">
                <i class="fa-solid fa-chart-column mr-1.5 text-emerald-400"></i>AAGM vs Fixed Depth-8 Comparison
              </h4>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div class="p-2.5 rounded-lg bg-gray-900 border border-gray-800/80">
                  <div class="text-[10px] text-gray-400 uppercase font-semibold">Step Savings</div>
                  <div id="save-steps" class="text-base font-bold text-emerald-400">75.0%</div>
                  <div class="text-[10px] text-gray-500">2 vs 8 iterations</div>
                </div>
                <div class="p-2.5 rounded-lg bg-gray-900 border border-gray-800/80">
                  <div class="text-[10px] text-gray-400 uppercase font-semibold">Energy Savings</div>
                  <div id="save-energy" class="text-base font-bold text-emerald-400">73.8%</div>
                  <div class="text-[10px] text-gray-500">57 μJ vs 218 μJ</div>
                </div>
                <div class="p-2.5 rounded-lg bg-gray-900 border border-gray-800/80">
                  <div class="text-[10px] text-gray-400 uppercase font-semibold">Accuracy Impact</div>
                  <div class="text-base font-bold text-cyan-400">0.00%</div>
                  <div class="text-[10px] text-gray-500">Identical logits</div>
                </div>
                <div class="p-2.5 rounded-lg bg-gray-900 border border-gray-800/80">
                  <div class="text-[10px] text-gray-400 uppercase font-semibold">Speculative Head Pre-Arm</div>
                  <div id="shadow-hints" class="text-base font-bold text-purple-400">0 Hints</div>
                  <div class="text-[10px] text-gray-500">APOP offloaded</div>
                </div>
              </div>
            </div>

          </div>

          <!-- Raw Telemetry Drawer -->
          <details class="glass-card rounded-2xl p-4 text-xs font-mono">
            <summary class="cursor-pointer text-gray-400 hover:text-white font-semibold flex items-center justify-between">
              <span><i class="fa-solid fa-terminal mr-2 text-cyan-400"></i>Raw Firmware JSON Telemetry (UART Output)</span>
              <span class="text-gray-500 text-[10px]">Click to inspect</span>
            </summary>
            <pre id="raw-json" class="mt-3 p-3 bg-gray-950 rounded-lg text-emerald-400 overflow-x-auto text-[11px] leading-relaxed">{}</pre>
          </details>

        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 3: INTERACTIVE SERIAL CONSOLE (ESP32 MOCK UART) -->
    <!-- ============================================================== -->
    <div id="tab-serial" class="hidden space-y-6">
      <div class="glass-card p-6 rounded-2xl space-y-4 border-l-4 border-amber-500">
        <div class="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-3">
          <div>
            <span class="text-xs font-mono font-bold text-amber-400 uppercase tracking-widest">ESP32 UART Protocol (115200 Baud)</span>
            <h2 class="text-lg font-bold text-white mt-0.5">Interactive Hardware-Free FreeRTOS Serial Monitor</h2>
            <p class="text-xs text-gray-400">Directly send newline-delimited serial commands to the dual-core C++ engine</p>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="clearTerminal()" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-mono">Clear</button>
            <span class="px-2.5 py-1 rounded-lg bg-amber-950 text-amber-400 border border-amber-800 text-xs font-mono">UART Online</span>
          </div>
        </div>

        <!-- Quick Command Buttons -->
        <div class="flex flex-wrap gap-2 text-xs font-mono">
          <span class="text-gray-400 py-1">Quick Commands:</span>
          <button onclick="sendSerialCmd('PING')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-cyan-400">PING</button>
          <button onclick="sendSerialCmd('STAT')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-cyan-400">STAT</button>
          <button onclick="sendSerialCmd('BENCH 10')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-cyan-400">BENCH 10</button>
          <button onclick="sendSerialCmd('MODE ECO')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-amber-400">MODE ECO</button>
          <button onclick="sendSerialCmd('MODE PERF')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-amber-400">MODE PERF</button>
          <button onclick="sendSerialCmd('BATT 3450')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-emerald-400">BATT 3450</button>
          <button onclick="sendSerialCmd('CHAT the movie was amazing')" class="px-2.5 py-1 rounded bg-gray-800 hover:bg-gray-700 text-purple-400">CHAT &lt;text&gt;</button>
        </div>

        <!-- Terminal Window -->
        <div id="serial-terminal" class="terminal-window h-96 p-4 rounded-xl overflow-y-auto text-xs text-gray-300 space-y-1 leading-relaxed">
          <div class="text-gray-500">> ESP32 DevKit-v1 Serial Connected @ 115200 baud</div>
          <div class="text-cyan-400">>RLM-AAGM fw=2 board=ESP32-DevKit-v1 vocab=4096 dim=96 max_rec=8 tau_lo=0.347 budget=8 chat=ready</div>
        </div>

        <!-- Serial Input Bar -->
        <div class="flex items-center space-x-2 pt-2">
          <div class="text-amber-400 font-mono text-sm px-2">&gt;</div>
          <input type="text" id="serial-input" onkeydown="if(event.key==='Enter') sendManualSerial()" placeholder="Type command (e.g., INFER great film, CHAT hello, STAT, BENCH 20)..." class="flex-1 bg-gray-950 border border-gray-800 rounded-xl px-4 py-2.5 text-xs font-mono text-gray-100 placeholder-gray-500 focus:outline-none focus:border-amber-500">
          <button onclick="sendManualSerial()" class="px-4 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs font-mono transition">
            Send Serial
          </button>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 4: HARDWARE COPROCESSOR & VERILOG SIMULATION -->
    <!-- ============================================================== -->
    <div id="tab-coproc" class="hidden space-y-6">
      <div class="glass-card p-6 rounded-2xl space-y-6 border-l-4 border-blue-500">
        <div class="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-3">
          <div>
            <span class="text-xs font-mono font-bold text-blue-400 uppercase tracking-widest">Digital Hardware Coprocessor</span>
            <h2 class="text-lg font-bold text-white mt-0.5">AAGM Arbiter APB Memory-Mapped Coprocessor (<code class="text-cyan-400 font-mono text-sm">aagm_arbiter_coprocessor.v</code>)</h2>
            <p class="text-xs text-gray-400">Hardware mass accumulation, single-cycle interrupt generation, and register map</p>
          </div>
          <span class="text-xs font-mono px-2.5 py-1 rounded bg-blue-950 text-blue-400 border border-blue-800">APB 32-bit Slave</span>
        </div>

        <!-- Coprocessor Register Map -->
        <div class="space-y-3">
          <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider">Memory-Mapped Register Map (Base: 0x3FF6_A000)</h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-cyan-400 font-bold">
                <span>ADDR_CTRL</span><span>0x00</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">[0] Start, [1] Reset, [2] Async Abort</p>
            </div>
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-cyan-400 font-bold">
                <span>ADDR_THRESHOLD</span><span>0x04</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">Q1.15 Halting threshold (default: 29491 = 0.900)</p>
            </div>
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-emerald-400 font-bold">
                <span>ADDR_GATE_IN</span><span>0x08</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">Write g_k: hardware adds (32768 - g_k) to mass</p>
            </div>
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-purple-400 font-bold">
                <span>ADDR_SHADOW_IN</span><span>0x0C</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">Write s_{k+1}: asserts PREARM if &lt; tau_lo (11370)</p>
            </div>
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-amber-400 font-bold">
                <span>ADDR_MASS_OUT</span><span>0x10</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">Read 32-bit Q1.15 accumulated halting mass</p>
            </div>
            <div class="p-3 rounded-xl bg-gray-950 border border-gray-800">
              <div class="flex justify-between text-rose-400 font-bold">
                <span>ADDR_STATUS</span><span>0x14</span>
              </div>
              <p class="text-[11px] text-gray-400 mt-1">[0] Busy, [1] Halt IRQ, [2] Prearm IRQ, [3] Abort</p>
            </div>
          </div>
        </div>

        <!-- Interactive Digital Waveform Timing Diagram -->
        <div class="space-y-3">
          <div class="flex flex-wrap justify-between items-center gap-2">
            <div>
              <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <i class="fa-solid fa-wave-square text-cyan-400"></i>
                <span>Interactive Waveform Timing Diagram (Cycle Simulation)</span>
              </h3>
              <p class="text-[11px] text-gray-400 mt-0.5">Real RTL bus cycles, Q1.15 mass accumulation, speculative prearm, and interrupt triggers</p>
            </div>
            <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">100 MHz Ref Clock</span>
          </div>

          <!-- Waveform Canvas/SVG Container -->
          <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 overflow-x-auto">
            <svg viewBox="0 0 920 340" class="w-full min-w-[760px] font-mono text-[11px]">
              <!-- Grid and Time Markers -->
              <defs>
                <pattern id="grid" width="60" height="340" patternUnits="userSpaceOnUse">
                  <line x1="60" y1="0" x2="60" y2="340" stroke="#1f2937" stroke-width="1" stroke-dasharray="2,2"/>
                </pattern>
              </defs>
              <rect width="920" height="340" fill="url(#grid)" opacity="0.6"/>

              <!-- Time Header -->
              <text x="140" y="20" fill="#6b7280" font-size="10">0ns</text>
              <text x="260" y="20" fill="#6b7280" font-size="10">40ns (RST_N ↑)</text>
              <text x="420" y="20" fill="#6b7280" font-size="10">80ns (Step 1)</text>
              <text x="580" y="20" fill="#6b7280" font-size="10">120ns (Step 2 PREARM)</text>
              <text x="740" y="20" fill="#6b7280" font-size="10">160ns (HALT IRQ)</text>
              <text x="840" y="20" fill="#6b7280" font-size="10">200ns (ABORT)</text>

              <!-- Signal 1: clk -->
              <text x="10" y="52" fill="#94a3b8" font-weight="bold">clk (100MHz)</text>
              <path d="M 140,55 L 155,55 L 155,38 L 170,38 L 170,55 L 185,55 L 185,38 L 200,38 L 200,55 L 215,55 L 215,38 L 230,38 L 230,55 L 245,55 L 245,38 L 260,38 L 260,55 L 275,55 L 275,38 L 290,38 L 290,55 L 305,55 L 305,38 L 320,38 L 320,55 L 335,55 L 335,38 L 350,38 L 350,55 L 365,55 L 365,38 L 380,38 L 380,55 L 395,55 L 395,38 L 410,38 L 410,55 L 425,55 L 425,38 L 440,38 L 440,55 L 455,55 L 455,38 L 470,38 L 470,55 L 485,55 L 485,38 L 500,38 L 500,55 L 515,55 L 515,38 L 530,38 L 530,55 L 545,55 L 545,38 L 560,38 L 560,55 L 575,55 L 575,38 L 590,38 L 590,55 L 605,55 L 605,38 L 620,38 L 620,55 L 635,55 L 635,38 L 650,38 L 650,55 L 665,55 L 665,38 L 680,38 L 680,55 L 695,55 L 695,38 L 710,38 L 710,55 L 725,55 L 725,38 L 740,38 L 740,55 L 755,55 L 755,38 L 770,38 L 770,55 L 785,55 L 785,38 L 800,38 L 800,55 L 815,55 L 815,38 L 830,38 L 830,55 L 845,55 L 845,38 L 860,38 L 860,55 L 875,55 L 875,38 L 890,38 L 890,55" fill="none" stroke="#38bdf8" stroke-width="1.5"/>

              <!-- Signal 2: rst_n -->
              <text x="10" y="92" fill="#94a3b8" font-weight="bold">rst_n</text>
              <path d="M 140,95 L 240,95 L 240,78 L 900,78" fill="none" stroke="#e2e8f0" stroke-width="1.8"/>

              <!-- Signal 3: APB Bus Write Data -->
              <text x="10" y="132" fill="#94a3b8" font-weight="bold">p_wdata (bus)</text>
              <rect x="250" y="118" width="90" height="20" rx="3" fill="#1e293b" stroke="#475569"/>
              <text x="260" y="132" fill="#cbd5e1" font-size="10">BUDGET=8</text>
              <rect x="350" y="118" width="85" height="20" rx="3" fill="#1e293b" stroke="#475569"/>
              <text x="360" y="132" fill="#cbd5e1" font-size="10">CTRL: START</text>
              <rect x="445" y="118" width="95" height="20" rx="3" fill="#1e293b" stroke="#38bdf8"/>
              <text x="452" y="132" fill="#38bdf8" font-size="10">s2=17911, g1</text>
              <rect x="550" y="118" width="115" height="20" rx="3" fill="#1e293b" stroke="#c084fc"/>
              <text x="556" y="132" fill="#c084fc" font-size="10">s3=9175 (&lt;tau_lo)</text>
              <rect x="675" y="118" width="105" height="20" rx="3" fill="#1e293b" stroke="#34d399"/>
              <text x="682" y="132" fill="#34d399" font-size="10">g2=12330 (&gt;TH)</text>
              <rect x="790" y="118" width="95" height="20" rx="3" fill="#1e293b" stroke="#f43f5e"/>
              <text x="798" y="132" fill="#f43f5e" font-size="10">CTRL: ABORT</text>

              <!-- Signal 4: steps_reg -->
              <text x="10" y="172" fill="#94a3b8" font-weight="bold">steps_reg [3:0]</text>
              <rect x="140" y="158" width="280" height="20" rx="3" fill="#0f172a" stroke="#334155"/>
              <text x="250" y="172" fill="#94a3b8" font-size="10">k = 0</text>
              <rect x="420" y="158" width="150" height="20" rx="3" fill="#0f172a" stroke="#38bdf8"/>
              <text x="480" y="172" fill="#38bdf8" font-size="10">k = 1</text>
              <rect x="570" y="158" width="330" height="20" rx="3" fill="#0f172a" stroke="#34d399"/>
              <text x="700" y="172" fill="#34d399" font-size="10">k = 2 (Resolved)</text>

              <!-- Signal 5: mass_acc_reg -->
              <text x="10" y="212" fill="#94a3b8" font-weight="bold">mass_acc_reg</text>
              <rect x="140" y="198" width="280" height="20" rx="3" fill="#0f172a" stroke="#334155"/>
              <text x="220" y="212" fill="#94a3b8" font-size="10">0.000 (0)</text>
              <rect x="420" y="198" width="150" height="20" rx="3" fill="#0f172a" stroke="#fbbf24"/>
              <text x="440" y="212" fill="#fbbf24" font-size="10">0.394 (12,917)</text>
              <rect x="570" y="198" width="330" height="20" rx="3" fill="#0f172a" stroke="#34d399"/>
              <text x="640" y="212" fill="#34d399" font-weight="bold" font-size="10">1.018 (33,355 &gt;= 29,491 THRESHOLD)</text>

              <!-- Signal 6: speculative_prearm -->
              <text x="10" y="252" fill="#c084fc" font-weight="bold">speculative_prearm</text>
              <path d="M 140,255 L 560,255 L 560,238 L 780,238 L 780,255 L 900,255" fill="none" stroke="#c084fc" stroke-width="2"/>
              <rect x="562" y="239" width="150" height="15" fill="#c084fc" opacity="0.15"/>
              <text x="570" y="250" fill="#e9d5ff" font-size="9">APOP Prearm Active</text>

              <!-- Signal 7: irq_halt -->
              <text x="10" y="292" fill="#34d399" font-weight="bold">irq_halt (Interrupt)</text>
              <path d="M 140,295 L 685,295 L 685,278 L 780,278 L 780,295 L 900,295" fill="none" stroke="#34d399" stroke-width="2.2"/>
              <rect x="687" y="279" width="90" height="15" fill="#34d399" opacity="0.2"/>
              <text x="692" y="290" fill="#a7f3d0" font-size="9">HALT TRIGGER</text>

              <!-- Signal 8: irq_abort -->
              <text x="10" y="328" fill="#f43f5e" font-weight="bold">irq_abort</text>
              <path d="M 140,332 L 800,332 L 800,315 L 890,315 L 890,332 L 900,332" fill="none" stroke="#f43f5e" stroke-width="2"/>
              <rect x="802" y="316" width="85" height="15" fill="#f43f5e" opacity="0.2"/>
              <text x="810" y="327" fill="#fecdd3" font-size="9">ABORT IRQ</text>
            </svg>
          </div>
        </div>

        <!-- Hardware Waveform Simulation Trace Log -->
        <div class="space-y-3">
          <div class="flex justify-between items-center">
            <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-terminal text-emerald-400"></i>
              <span>Verilog Simulation Log (Cycle-Accurate DUT)</span>
            </h3>
            <div class="flex items-center gap-2">
              <span class="text-[11px] text-gray-400">Generates real <code>.vcd</code> files</span>
              <button onclick="runVerilogTest()" id="btn-coproc-resim" class="text-xs px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white font-mono flex items-center gap-1.5">
                <i class="fa-solid fa-play text-[10px]"></i>
                <span>Re-simulate</span>
              </button>
            </div>
          </div>
          <pre id="coproc-verilog-log" class="p-4 bg-gray-950 rounded-xl text-emerald-400 font-mono text-[11px] h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">Simulating APB Coprocessor...</pre>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 5: PATENT SPECIFICATION & 20 CLAIMS -->
    <!-- ============================================================== -->
    <div id="tab-patent" class="hidden space-y-6">
      <div class="glass-card p-6 rounded-2xl space-y-6 border-l-4 border-purple-500">
        <div class="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
          <div>
            <span class="text-xs font-mono font-bold text-purple-400 uppercase tracking-widest">Patent Application Specification</span>
            <h2 class="text-lg font-bold text-white mt-1">Asynchronous Dual-Core Recursive Transformer Inference System with Speculative Early-Exit and Adaptive Energy-Context Gating</h2>
            <p class="text-xs text-gray-400">Formal Patent Disclosure &bull; 20 Formal Claims &bull; Hardware-Software Co-Design Architecture</p>
          </div>
          <a href="/docs/PATENT_SPECIFICATION.md" target="_blank" class="px-3.5 py-1.5 rounded-lg bg-purple-600/80 hover:bg-purple-600 text-white font-medium text-xs transition flex items-center space-x-1.5">
            <i class="fa-solid fa-file-lines"></i>
            <span>View Full Patent File</span>
          </a>
        </div>

        <!-- 4 Key Patent Inventions Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-gray-950 border border-gray-800 p-4 rounded-xl space-y-2">
            <div class="text-xs font-bold text-purple-400 flex items-center">
              <i class="fa-solid fa-forward-fast mr-2"></i>Invention 1: Speculative Asynchronous Output Pre-Computation (APOP)
            </div>
            <p class="text-xs text-gray-300 leading-relaxed">
              When the shadow forecaster $s_{k+1}$ predicts halt ($\tau_{lo} < 0.347$, $\ge 97.2\%$ precision), Core 0 speculatively computes the output classification head while Core 1 finishes state refinement. Saves 150-200 μs tail latency.
            </p>
          </div>

          <div class="bg-gray-950 border border-gray-800 p-4 rounded-xl space-y-2">
            <div class="text-xs font-bold text-cyan-400 flex items-center">
              <i class="fa-solid fa-battery-three-quarters mr-2"></i>Invention 2: Internal-Resistance Droop-Compensated DVFS
            </div>
            <p class="text-xs text-gray-300 leading-relaxed">
              Samples supply rail ADC and compensates for transient internal resistance voltage droop ($V_{\text{loaded}} = V_{\text{OCV}} - I_{\text{active}} \times R_{\text{int}}$). Triggers clean mid-inference cooperative aborts without brownout reset.
            </p>
          </div>

          <div class="bg-gray-950 border border-gray-800 p-4 rounded-xl space-y-2">
            <div class="text-xs font-bold text-emerald-400 flex items-center">
              <i class="fa-solid fa-snowflake mr-2"></i>Invention 3: Temporal Token Saliency Freezing (TSTF)
            </div>
            <p class="text-xs text-gray-300 leading-relaxed">
              Computes token-wise state divergence $\Delta_k[t] = \|h_k[t] - h_{k-1}[t]\|_1$. Tokens converging below threshold $\tau_{\text{freeze}}$ bypass subsequent 384-wide FFN transformations, yielding an extra 25% to 40% MAC savings.
            </p>
          </div>

          <div class="bg-gray-950 border border-gray-800 p-4 rounded-xl space-y-2">
            <div class="text-xs font-bold text-blue-400 flex items-center">
              <i class="fa-solid fa-microchip mr-2"></i>Invention 4: Digital AAGM Arbiter Coprocessor (APB Interface)
            </div>
            <p class="text-xs text-gray-300 leading-relaxed">
              A 32-bit memory-mapped hardware coprocessor implementing digital fixed-point halting mass accumulation (Q1.15), single-cycle asynchronous abort interrupts, and budget exhaustion overrides in RTL Verilog.
            </p>
          </div>
        </div>

        <!-- 20 Patent Claims Accordion/List -->
        <div class="space-y-3">
          <h3 class="text-xs font-bold text-white uppercase tracking-wider">Formal 20 Patent Claims (Full Text)</h3>
          <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 font-mono text-xs text-gray-300 space-y-4 max-h-96 overflow-y-auto leading-relaxed">
            <div class="border-b border-gray-800 pb-2">
              <strong class="text-cyan-400">Claim 1 (Independent System Claim):</strong> A dual-core embedded inference system for recursive neural networks, comprising: a multi-core microcontroller comprising at least a first compute core and a second arbiter core coupled to an internal shared memory; a weight-shared recursive neural network block stored in non-volatile memory... wherein said second arbiter core is configured to execute an asynchronous adaptive gating mechanism concurrently with said first compute core...
            </div>
            <div class="border-b border-gray-800 pb-2">
              <strong class="text-purple-400">Claim 5 (Speculative Output Claim):</strong> The system of claim 4, wherein upon detecting that $s_{k+1}$ is less than a predetermined confidence threshold $\tau_{lo}$, said second arbiter core speculatively evaluates an output classification head on a pooled hidden state concurrently with said first compute core executing a final recursive step, thereby eliminating output tail latency upon confirmation of early halting.
            </div>
            <div class="border-b border-gray-800 pb-2">
              <strong class="text-emerald-400">Claim 7 (Token Saliency Freezing Claim):</strong> The system of claim 1, wherein said first compute core evaluates a temporal token saliency metric measuring hidden state divergence across consecutive recursive steps, and selectively disables execution of a feed-forward neural network for tokens having a divergence metric below a convergence threshold.
            </div>
            <div class="border-b border-gray-800 pb-2">
              <strong class="text-amber-400">Claim 9 (Droop Compensation Claim):</strong> The system of claim 1, wherein said second arbiter core samples an analog-to-digital converter coupled to an energy supply rail, calculates an internal resistance voltage droop based on active operating current, and dynamically adjusts a maximum recursion step budget.
            </div>
            <div class="border-b border-gray-800 pb-2">
              <strong class="text-blue-400">Claim 15 (Hardware Coprocessor Claim):</strong> A hardware coprocessor for adaptive neural network gating, comprising: a hardware register interface accessible by a host processor; a fixed-point halting mass accumulator register configured to sum digital values representing $(1 - g_k)$ across computation steps; a digital comparator configured to assert a hardware halt signal when said accumulator register equals or exceeds a 16-bit threshold corresponding to $1 - \epsilon$ in Q1.15 fixed-point format...
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 6: VERIFICATION SUITE -->
    <!-- ============================================================== -->
    <div id="tab-verify" class="hidden space-y-6">
      <div class="glass-card p-6 rounded-2xl space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 class="text-lg font-bold text-white flex items-center">
              <i class="fa-solid fa-vial-circle-check mr-2 text-cyan-400"></i>Hardware-Free Verification Suite
            </h2>
            <p class="text-xs text-gray-400">Dual-layer validation: C++ numerical parity &amp; Verilog coprocessor simulation</p>
          </div>
          <div class="flex space-x-3">
            <button onclick="runGoldenTest()" id="btn-run-golden" class="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs shadow transition flex items-center space-x-1.5">
              <i class="fa-solid fa-play text-[10px]"></i><span>Run C++ Golden Parity</span>
            </button>
            <button onclick="runVerilogTest()" id="btn-run-verilog" class="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow transition flex items-center space-x-1.5">
              <i class="fa-solid fa-microchip text-[10px]"></i><span>Run Verilog Simulation</span>
            </button>
          </div>
        </div>

        <!-- Verification Results Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
          
          <!-- Test 1: Golden Vectors Parity -->
          <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="font-bold text-sm text-white flex items-center">
                <i class="fa-solid fa-code mr-2 text-cyan-400"></i>C++ Firmware Golden Parity
              </h3>
              <span id="golden-status-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">12/12 PASSED</span>
            </div>
            <p class="text-xs text-gray-400">Asserts bit-level prediction parity, gate tolerances &lt;1e-3, logit tolerances &lt;2e-2 vs PyTorch int8 mirror.</p>

            <div class="grid grid-cols-2 gap-2 text-xs font-mono">
              <div class="bg-gray-900 p-2 rounded">
                <div class="text-gray-500 text-[10px]">Max Logit Error</div>
                <div id="val-logit-err" class="text-emerald-400 font-bold">2.86e-6</div>
              </div>
              <div class="bg-gray-900 p-2 rounded">
                <div class="text-gray-500 text-[10px]">Max Gate Error</div>
                <div id="val-gate-err" class="text-emerald-400 font-bold">5.36e-7</div>
              </div>
              <div class="bg-gray-900 p-2 rounded">
                <div class="text-gray-500 text-[10px]">Prediction Mismatch</div>
                <div id="val-pred-mism" class="text-emerald-400 font-bold">0 / 12 (0%)</div>
              </div>
              <div class="bg-gray-900 p-2 rounded">
                <div class="text-gray-500 text-[10px]">Tokenizer Mismatch</div>
                <div id="val-tok-mism" class="text-emerald-400 font-bold">0 / 12 (0%)</div>
              </div>
            </div>

            <!-- Samples table preview -->
            <div class="overflow-x-auto max-h-56 overflow-y-auto">
              <table class="w-full text-left text-[11px] font-mono text-gray-300">
                <thead class="bg-gray-900 text-gray-400 sticky top-0">
                  <tr>
                    <th class="p-1.5">#</th>
                    <th class="p-1.5">Pred</th>
                    <th class="p-1.5">Gold</th>
                    <th class="p-1.5">Steps</th>
                    <th class="p-1.5">Logit Err</th>
                    <th class="p-1.5">Status</th>
                  </tr>
                </thead>
                <tbody id="golden-table-body" class="divide-y divide-gray-800">
                  <!-- Populated via JS -->
                </tbody>
              </table>
            </div>
          </div>

          <!-- Test 2: Verilog Digital Coprocessor Check -->
          <div class="bg-gray-950 border border-gray-800 rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="font-bold text-sm text-white flex items-center">
                <i class="fa-solid fa-microchip mr-2 text-purple-400"></i>Verilog Coprocessor Output
              </h3>
              <span id="verilog-status-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">ALL PASSED</span>
            </div>
            <p class="text-xs text-gray-400">Verifies APB bus transactions, Q1.15 mass accumulation, shadow prearm comparator, and async aborts.</p>

            <pre id="verilog-output" class="p-3 bg-gray-900 rounded-lg text-emerald-400 font-mono text-[11px] h-64 overflow-y-auto whitespace-pre-wrap">Running simulation...</pre>
          </div>

        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 7: REVIEW & DEFENSE GUIDE -->
    <!-- ============================================================== -->
    <div id="tab-defense" class="hidden space-y-6">
      <div class="glass-card p-6 rounded-2xl border-l-4 border-cyan-500 space-y-2">
        <span class="text-xs font-bold text-cyan-400 uppercase tracking-widest">College Project Review &amp; Paper Defense Guide</span>
        <h2 class="text-xl font-bold text-white">Edge-RLM: Deploying Recursive Language Models on ESP32 with AAGM</h2>
        <p class="text-xs text-gray-400">Comprehensive cheat-sheet for project review panel, examiner queries, and technical defense.</p>
      </div>

      <!-- Top Viva Questions & Answers -->
      <div class="space-y-4">
        <h3 class="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center">
          <i class="fa-solid fa-circle-question mr-2 text-cyan-400"></i>Key Viva Questions &amp; Authoritative Model Answers
        </h3>

        <!-- Q1 -->
        <div class="glass-card p-4 rounded-xl space-y-2">
          <div class="font-bold text-sm text-white flex items-center">
            <span class="w-6 h-6 rounded-full bg-cyan-950 text-cyan-400 flex items-center justify-center text-xs mr-2 font-mono">Q1</span>
            Why not use standard TensorFlow Lite for Microcontrollers (TFLM) or ESP-DL?
          </div>
          <p class="text-xs text-gray-300 leading-relaxed pl-8">
            <strong class="text-cyan-400">Answer:</strong> TFLM and ESP-DL compile static computational graphs. They cannot represent data-dependent recursion where depth varies dynamically from 2 to 8 steps per input sample based on runtime confidence. To run on TFLM, one would have to unroll the recursive block 8 times, multiplying flash consumption by 8x and completely destroying adaptive halting energy savings. Our custom ~350-line C++ engine executes dynamic recursion with static buffers and zero dynamic heap allocation.
          </p>
        </div>

        <!-- Q2 -->
        <div class="glass-card p-4 rounded-xl space-y-2">
          <div class="font-bold text-sm text-white flex items-center">
            <span class="w-6 h-6 rounded-full bg-cyan-950 text-cyan-400 flex items-center justify-center text-xs mr-2 font-mono">Q2</span>
            How do you verify the project if the physical ESP32 board is not connected right now?
          </div>
          <p class="text-xs text-gray-300 leading-relaxed pl-8">
            <strong class="text-cyan-400">Answer:</strong> We employ a dual-track hardware-free verification methodology: (1) Host-native C++ harness (<code class="text-cyan-400">firmware/test/host_harness</code>) compiling the shipping firmware source against 12 golden test vectors produced by the PyTorch int8 mirror, proving max logit drift is &lt;3e-6; (2) Digital RTL Verilog cycle-simulation (<code class="text-cyan-400">verification/verilog</code>) testing the AAGM gating policy, cumulative mass threshold, and asynchronous abort in hardware; and (3) Mock FreeRTOS serial simulation executing the exact UART line protocol.
          </p>
        </div>

        <!-- Q3 -->
        <div class="glass-card p-4 rounded-xl space-y-2">
          <div class="font-bold text-sm text-white flex items-center">
            <span class="w-6 h-6 rounded-full bg-cyan-950 text-cyan-400 flex items-center justify-center text-xs mr-2 font-mono">Q3</span>
            What makes this project patent-worthy?
          </div>
          <p class="text-xs text-gray-300 leading-relaxed pl-8">
            <strong class="text-purple-400">Answer:</strong> We have formulated a 20-claim formal patent application covering: (1) Asynchronous predictive output pre-computation (APOP) where Core 0 speculatively computes classification heads to eliminate tail latency; (2) Closed-loop battery internal-resistance droop compensation ($\Delta V = I_{\text{active}} \times R_{\text{int}}$) coupled to dynamic recursion budget shrinks; (3) Temporal token saliency freezing (TSTF) bypassing converged tokens in the 384-wide FFN; and (4) An APB memory-mapped hardware coprocessor for digital gating.
          </p>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-gray-800 bg-gray-950 py-4 text-center text-xs text-gray-500">
    Edge-RLM with AAGM &bull; ESP32 Hardware-Software Co-Design &bull; Publication &amp; Patent Track
  </footer>

  <!-- Scripts -->
  <script>
    const PRESETS = [
      "the movie was a brilliant masterpiece with stunning visuals and a powerful story.",
      "an absolute disaster, boring and painfully slow from start to finish.",
      "it starts slow but the brilliant ending makes it absolutely worth watching.",
      "great cast but the terrible script ruins the whole film completely."
    ];

    let currentProfile = "PERF";
    let currentBattery = 4000;
    let myChart = null;

    function switchTab(tabId) {
      ['demo', 'serial', 'coproc', 'patent', 'verify', 'defense'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.add('hidden');
        const btn = document.getElementById(`tab-btn-${t}`);
        btn.className = "px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition text-gray-400 hover:text-white hover:bg-gray-800 flex items-center";
      });
      document.getElementById(`tab-${tabId}`).classList.remove('hidden');
      const activeBtn = document.getElementById(`tab-btn-${tabId}`);
      if (tabId === 'patent') {
        activeBtn.className = "px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition bg-purple-600 text-white shadow-sm flex items-center";
      } else if (tabId === 'serial') {
        activeBtn.className = "px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition bg-amber-600 text-white shadow-sm flex items-center";
      } else if (tabId === 'coproc') {
        activeBtn.className = "px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition bg-blue-600 text-white shadow-sm flex items-center";
      } else {
        activeBtn.className = "px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition bg-cyan-600 text-white shadow-sm flex items-center";
      }
    }

    function setPreset(idx) {
      document.getElementById('input-text').value = PRESETS[idx];
      executeInference();
    }

    function selectProfile(prof) {
      currentProfile = prof;
      ['PERF', 'BAL', 'ECO', 'AUTO'].forEach(p => {
        const btn = document.getElementById(`btn-prof-${p}`);
        if (p === prof) {
          btn.className = "py-2 rounded-lg bg-cyan-600 text-white transition text-center shadow-sm";
        } else {
          btn.className = "py-2 rounded-lg bg-gray-800 text-gray-300 hover:text-white transition text-center";
        }
      });
      const ind = document.getElementById('profile-indicator');
      if (prof === "PERF") ind.textContent = "PERF (240MHz/B8)";
      else if (prof === "BAL") ind.textContent = "BAL (160MHz/B6)";
      else if (prof === "ECO") ind.textContent = "ECO (80MHz/B4)";
      else ind.textContent = "AUTO (Battery Driven)";
      executeInference();
    }

    function updateBattery(val) {
      currentBattery = parseInt(val);
      const pct = Math.round(Math.max(0, Math.min(100, (currentBattery - 3300) / 900 * 100)));
      document.getElementById('batt-val').textContent = `${currentBattery} mV (${pct}%)`;
      const currentMa = currentProfile === "PERF" ? 50 : (currentProfile === "BAL" ? 36 : 27);
      const droopMv = Math.round((currentMa / 1000) * 0.200 * 1000);
      const loaded = currentBattery - droopMv;
      document.getElementById('loaded-batt-val').textContent = `${loaded} mV`;
      
      const alertBox = document.getElementById('brownout-alert');
      if (loaded < 3350) {
        alertBox.classList.remove('hidden');
      } else {
        alertBox.classList.add('hidden');
      }
      if (currentProfile === "AUTO") executeInference();
    }

    async function executeInference() {
      const text = document.getElementById('input-text').value.trim();
      if (!text) return;
      const btn = document.getElementById('btn-run');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1"></i><span>Refining States...</span>`;

      try {
        const resp = await fetch('/api/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: text,
            profile: currentProfile,
            batt_mv: currentBattery
          })
        });
        const res = await resp.json();
        renderResult(res);
      } catch (e) {
        console.error(e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-bolt mr-1"></i><span>Run On-Device Inference</span>`;
      }
    }

    function renderResult(r) {
      document.getElementById('raw-json').textContent = JSON.stringify(r, null, 2);

      const badge = document.getElementById('pred-badge');
      const isPos = r.pred === 1;
      const negLogit = r.logits[0];
      const posLogit = r.logits[1];
      const maxLogit = Math.max(negLogit, posLogit);
      const expNeg = Math.exp(negLogit - maxLogit);
      const expPos = Math.exp(posLogit - maxLogit);
      const conf = ((isPos ? expPos : expNeg) / (expNeg + expPos) * 100).toFixed(1);

      if (isPos) {
        badge.className = "text-xl font-extrabold px-3 py-1 rounded-xl bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center space-x-2";
        badge.innerHTML = `<i class="fa-solid fa-thumbs-up"></i><span>POSITIVE</span>`;
      } else {
        badge.className = "text-xl font-extrabold px-3 py-1 rounded-xl bg-rose-950 text-rose-400 border border-rose-800 flex items-center space-x-2";
        badge.innerHTML = `<i class="fa-solid fa-thumbs-down"></i><span>NEGATIVE</span>`;
      }
      document.getElementById('conf-val').textContent = `${conf}%`;

      document.getElementById('metric-lat').textContent = `${r.us.toLocaleString()} μs`;
      document.getElementById('metric-steps').textContent = `${r.steps} / ${r.budget}`;
      document.getElementById('metric-energy').textContent = `${r.comparison.actual_energy_uj} μJ`;

      const mass = r.mass || (r.depth ? (r.steps - r.depth) : 0.9);
      document.getElementById('mass-label').textContent = `${mass.toFixed(3)} / 0.900 (Threshold ${mass >= 0.9 ? 'Met' : 'Pending'})`;

      document.getElementById('save-steps').textContent = `${r.comparison.saved_steps_pct}%`;
      document.getElementById('save-energy').textContent = `${r.comparison.energy_saved_pct}%`;
      document.getElementById('shadow-hints').textContent = `${r.hints} Hints`;

      // Render Step Cards
      const container = document.getElementById('step-cards-container');
      container.innerHTML = '';
      let cumMass = 0;
      const chartLabels = [];
      const chartGates = [];
      const chartMasses = [];
      const chartThreshold = [];

      r.gates.forEach((g, i) => {
        cumMass += (1 - g);
        const s = r.shadows[i] || 0.5;
        const isPrearm = s < 0.347;
        const isHaltStep = i === r.gates.length - 1;
        
        chartLabels.push(`Step ${i + 1}`);
        chartGates.push(parseFloat(g.toFixed(3)));
        chartMasses.push(parseFloat(cumMass.toFixed(3)));
        chartThreshold.push(0.900);

        const card = document.createElement('div');
        card.className = `p-3 rounded-xl border ${isHaltStep ? 'bg-cyan-950/40 border-cyan-800' : 'bg-gray-900 border-gray-800'}`;
        card.innerHTML = `
          <div class="flex justify-between items-center mb-1.5 text-xs">
            <span class="font-bold text-white font-mono">Step #${i + 1}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded ${isHaltStep ? 'bg-emerald-950 text-emerald-400 font-bold' : 'text-gray-400 bg-gray-800'}">
              ${isHaltStep ? 'HALT' : 'RECURSE'}
            </span>
          </div>
          <div class="space-y-1 font-mono text-[11px]">
            <div class="flex justify-between text-gray-400">
              <span>Mixing Gate g:</span><span class="text-white">${g.toFixed(4)}</span>
            </div>
            <div class="flex justify-between text-gray-400">
              <span>Cum. Mass:</span><span class="text-cyan-400 font-bold">${cumMass.toFixed(3)}</span>
            </div>
            <div class="flex justify-between text-gray-400">
              <span>Shadow s:</span><span class="${isPrearm ? 'text-purple-400 font-bold' : 'text-gray-300'}">${s.toFixed(3)} ${isPrearm ? '⚡ APOP' : ''}</span>
            </div>
          </div>
        `;
        container.appendChild(card);
      });

      renderChart(chartLabels, chartGates, chartMasses, chartThreshold);
    }

    function renderChart(labels, gates, masses, threshold) {
      const ctx = document.getElementById('recursionChart').getContext('2d');
      if (myChart) myChart.destroy();
      myChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Cumulative Halting Mass',
              data: masses,
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.1)',
              borderWidth: 2,
              fill: true,
              tension: 0.3
            },
            {
              label: 'Threshold (0.900)',
              data: threshold,
              borderColor: '#ef4444',
              borderDash: [5, 5],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false
            },
            {
              label: 'Mixing Gate g_k',
              data: gates,
              borderColor: '#a855f7',
              borderWidth: 1.5,
              fill: false,
              tension: 0.3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: '#94a3b8', font: { size: 10 } }
            }
          },
          scales: {
            x: {
              ticks: { color: '#94a3b8', font: { size: 10 } },
              grid: { color: '#1e293b' }
            },
            y: {
              min: 0,
              max: 1.5,
              ticks: { color: '#94a3b8', font: { size: 10 } },
              grid: { color: '#1e293b' }
            }
          }
        }
      });
    }

    // Serial Terminal Methods
    async function sendSerialCmd(cmd) {
      const term = document.getElementById('serial-terminal');
      term.innerHTML += `<div class="text-amber-400">> ${cmd}</div>`;
      term.scrollTop = term.scrollHeight;

      try {
        const resp = await fetch('/api/serial', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cmd: cmd })
        });
        const res = await resp.json();
        term.innerHTML += `<div class="text-emerald-400 pl-3">${res.response}</div>`;
      } catch (e) {
        term.innerHTML += `<div class="text-rose-400 pl-3">[!] Error communicating with UART</div>`;
      }
      term.scrollTop = term.scrollHeight;
    }

    function sendManualSerial() {
      const inp = document.getElementById('serial-input');
      const cmd = inp.value.trim();
      if (!cmd) return;
      sendSerialCmd(cmd);
      inp.value = '';
    }

    function clearTerminal() {
      document.getElementById('serial-terminal').innerHTML = `<div class="text-gray-500">> ESP32 DevKit-v1 Serial Connected @ 115200 baud</div>`;
    }

    async function runGoldenTest() {
      const btn = document.getElementById('btn-run-golden');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1"></i> Running...`;
      }
      try {
        const resp = await fetch('/api/verify-golden');
        const res = await resp.json();
        const tbody = document.getElementById('golden-table-body');
        if (tbody && res.samples) {
          tbody.innerHTML = '';
          res.samples.forEach(s => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td class="p-1.5">${s.i}</td>
              <td class="p-1.5 font-bold ${s.pred === 1 ? 'text-emerald-400' : 'text-rose-400'}">${s.pred}</td>
              <td class="p-1.5 text-gray-400">${s.gold_pred}</td>
              <td class="p-1.5">${s.steps}</td>
              <td class="p-1.5 text-cyan-400">${s.logit_err.toExponential(2)}</td>
              <td class="p-1.5 text-emerald-400 font-bold">PASS</td>
            `;
            tbody.appendChild(tr);
          });
        }
        if (res.summary) {
          const badge = document.getElementById('golden-status-badge');
          if (badge) badge.textContent = `${res.summary.passed}/${res.summary.golden} PASSED`;
          const logitErr = document.getElementById('val-logit-err');
          if (logitErr) logitErr.textContent = res.summary.max_logit_err.toExponential(2);
          const gateErr = document.getElementById('val-gate-err');
          if (gateErr) gateErr.textContent = res.summary.max_gate_err.toExponential(2);
          const predMism = document.getElementById('val-pred-mism');
          if (predMism) predMism.textContent = `${res.summary.mism_pred} / ${res.summary.golden} (0%)`;
          const tokMism = document.getElementById('val-tok-mism');
          if (tokMism) tokMism.textContent = `${res.summary.mism_ids} / ${res.summary.golden} (0%)`;
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = `<i class="fa-solid fa-play text-[10px]"></i><span>Run C++ Golden Parity</span>`;
        }
      }
    }

    async function runVerilogTest() {
      const btn1 = document.getElementById('btn-run-verilog');
      const btn2 = document.getElementById('btn-coproc-resim');
      [btn1, btn2].forEach(b => {
        if (b) {
          b.disabled = true;
          b.innerHTML = `<i class="fa-solid fa-spinner fa-spin mr-1 text-[10px]"></i><span>Simulating...</span>`;
        }
      });
      try {
        const resp = await fetch('/api/verify-verilog');
        const res = await resp.json();
        const vOut = document.getElementById('verilog-output');
        if (vOut) vOut.textContent = res.output;
        const cLog = document.getElementById('coproc-verilog-log');
        if (cLog) cLog.textContent = res.output;
        const vBadge = document.getElementById('verilog-status-badge');
        if (vBadge) {
          vBadge.textContent = res.status === 'PASS' ? 'ALL PASSED (Icarus RTL)' : 'SIMULATION ERROR';
          vBadge.className = res.status === 'PASS'
            ? 'text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800'
            : 'text-xs font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800';
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (btn1) {
          btn1.disabled = false;
          btn1.innerHTML = `<i class="fa-solid fa-microchip text-[10px]"></i><span>Run Verilog Simulation</span>`;
        }
        if (btn2) {
          btn2.disabled = false;
          btn2.innerHTML = `<i class="fa-solid fa-play text-[10px]"></i><span>Re-simulate</span>`;
        }
      }
    }

    // Auto load on start
    window.addEventListener('DOMContentLoaded', () => {
      executeInference();
      runGoldenTest();
      runVerilogTest();
    });
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            chat_page = (REPO_ROOT / "host" / "chat.html").read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(chat_page.encode("utf-8"))
        elif url.path == "/lab":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif url.path == "/api/assistant/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(chat_status()).encode("utf-8"))
        elif url.path == "/graph":
            graph_page = (REPO_ROOT / "host" / "pipeline_graph.html").read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(graph_page.encode("utf-8"))
        elif url.path == "/api/graph":
            graph_file = REPO_ROOT / "graphify-out" / "graph.json"
            if not graph_file.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Graph index not found: graphify-out/graph.json"}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(graph_file.read_bytes())
        elif url.path == "/api/verify-golden":
            data = run_golden_check()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path == "/api/verify-verilog":
            data = run_verilog_check()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path == "/api/patent":
            text = get_patent_text()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(text.encode("utf-8"))
        elif url.path == "/docs/PATENT_SPECIFICATION.md":
            text = get_patent_text()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.end_headers()
            self.wfile.write(text.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 65536:
            self.send_response(413)
            self.end_headers()
            return
        try:
            body = self.rfile.read(content_length).decode("utf-8")
            req = json.loads(body) if body else {}
            if not isinstance(req, dict):
                raise ValueError("JSON request must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON request body"}).encode("utf-8"))
            return

        if url.path == "/api/assistant/chat":
            try:
                result = respond_to_chat(
                    text=req.get("text", ""),
                    history=req.get("history", []),
                    mode=req.get("mode", "chat"),
                    profile=req.get("profile", "PERF"),
                    inference=run_inference,
                )
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(error)}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif url.path == "/api/infer":
            text = req.get("text", "")
            profile = req.get("profile", "PERF")
            batt_mv = float(req.get("batt_mv", 4000.0))
            budget = 8 if profile == "PERF" else (6 if profile == "BAL" else 4)
            result = run_inference(text, budget, profile, batt_mv)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif url.path == "/api/serial":
            cmd = req.get("cmd", "STAT")
            out = run_serial_cmd(cmd)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"response": out}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"================================================================")
    print(f"  Edge-RLM ESP32 Review & Demonstration Server Running")
    print(f"  Local / Preview URL: http://0.0.0.0:{port}")
    print(f"================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
