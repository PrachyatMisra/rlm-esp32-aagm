# Hardware-free verification

The project has two complementary checks:

1. `firmware/test` compiles the actual shipping ESP32 C++ engine on macOS and
   compares it with `tools/out/golden_vectors.txt`. This is the authoritative
   software parity check for tokenizer, attention, logits, gates, shadows,
   halting, and predictions.
2. `verification/verilog` uses the free Icarus Verilog simulator to check the
   small digital AAGM policy around the neural engine: minimum recursion,
   cumulative halting threshold, budget exhaustion, and asynchronous abort.

The Verilog module does not simulate the ESP32 Xtensa CPU or the floating-point
neural network. Use the native harness for model/firmware numerical parity.

## macOS commands

From the repository root:

```bash
make -C firmware/test check
make -C verification/verilog check
python host/gateway.py --selftest
```

Install Icarus Verilog with Homebrew if needed:

```bash
brew install icarus-verilog
```

The native check requires a C++ compiler. The Python gateway self-test also
requires the generated golden vectors and does not require PyTorch or a board.

## Optional model regeneration

To retrain and regenerate weights and golden vectors:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/train_export.py --dataset synthetic --fast --epochs 1 --shadow-epochs 1
make -C firmware/test check
```

Regenerated headers are under `firmware/rlm_esp32/src/`; flash the board again
after regeneration.
