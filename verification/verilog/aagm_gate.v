// Hardware-free reference of the digital AAGM control policy.
// The neural network itself remains in portable C++/float32; this module
// verifies the deterministic budget, halt, and abort decisions around it.
module aagm_gate (
    input  wire       step_valid,
    input  wire       min_steps_reached,
    input  wire       budget_exhausted,
    input  wire       async_abort,
    input  wire [15:0] halting_mass_q15,
    output wire       halt,
    output wire       abort
);
    // 0.9 in unsigned Q1.15 fixed point, matching RLM_HALT_EPS = 0.1.
    assign halt = step_valid && min_steps_reached &&
                  !budget_exhausted && (halting_mass_q15 >= 16'd29491);
    assign abort = step_valid && async_abort;
endmodule
