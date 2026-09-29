IVERILOG ?= iverilog
VVP ?= vvp

all: check

aagm_gate.vvp: aagm_gate.v tb_aagm_gate.v
	$(IVERILOG) -g2012 -o $@ aagm_gate.v tb_aagm_gate.v

check: aagm_gate.vvp
	$(VVP) aagm_gate.vvp

clean:
	rm -f aagm_gate.vvp

.PHONY: all check clean
