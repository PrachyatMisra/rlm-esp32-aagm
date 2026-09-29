IVERILOG ?= $(shell which iverilog 2>/dev/null)
VVP ?= $(shell which vvp 2>/dev/null)

all: check

check: check-gate check-coprocessor

check-gate:
	@if [ -n "$(IVERILOG)" ] && [ -n "$(VVP)" ]; then \
		echo "Testing AAGM Gate (Icarus Verilog)..."; \
		$(IVERILOG) -g2012 -o aagm_gate.vvp aagm_gate.v tb_aagm_gate.v && \
		$(VVP) aagm_gate.vvp; \
	else \
		echo "Testing AAGM Gate (Reference Simulator)..."; \
		python3 sim_aagm_gate.py; \
	fi

check-coprocessor:
	@if [ -n "$(IVERILOG)" ] && [ -n "$(VVP)" ]; then \
		echo "Testing AAGM Arbiter Coprocessor (Icarus Verilog)..."; \
		$(IVERILOG) -g2012 -o aagm_coprocessor.vvp aagm_arbiter_coprocessor.v tb_aagm_arbiter_coprocessor.v && \
		$(VVP) aagm_coprocessor.vvp; \
	else \
		echo "Testing AAGM Arbiter Coprocessor (Reference Simulator)..."; \
		python3 sim_aagm_arbiter_coprocessor.py; \
	fi

clean:
	rm -f aagm_gate.vvp aagm_gate.vcd aagm_coprocessor.vvp aagm_coprocessor.vcd

.PHONY: all check check-gate check-coprocessor clean
