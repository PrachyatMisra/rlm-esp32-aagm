`timescale 1ns / 1ps

module tb_aagm_gate;
    reg step_valid;
    reg min_steps_reached;
    reg budget_exhausted;
    reg async_abort;
    reg [15:0] halting_mass_q15;
    wire halt;
    wire abort;
    integer failures;

    aagm_gate dut (
        .step_valid(step_valid),
        .min_steps_reached(min_steps_reached),
        .budget_exhausted(budget_exhausted),
        .async_abort(async_abort),
        .halting_mass_q15(halting_mass_q15),
        .halt(halt),
        .abort(abort)
    );

    task check_outputs;
        input expected_halt;
        input expected_abort;
        begin
            #1;
            if (halt !== expected_halt || abort !== expected_abort) begin
                $display("FAIL halt=%b abort=%b expected=%b/%b", halt, abort,
                         expected_halt, expected_abort);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("aagm_gate.vcd");
        $dumpvars(0, tb_aagm_gate);
        failures = 0;
        step_valid = 0; min_steps_reached = 1; budget_exhausted = 0;
        async_abort = 0; halting_mass_q15 = 16'd30000;
        check_outputs(0, 0);

        step_valid = 1; min_steps_reached = 0;
        check_outputs(0, 0);
        min_steps_reached = 1; halting_mass_q15 = 16'd29490;
        check_outputs(0, 0);
        halting_mass_q15 = 16'd29491;
        check_outputs(1, 0);
        budget_exhausted = 1;
        check_outputs(0, 0);
        budget_exhausted = 0; async_abort = 1;
        check_outputs(1, 1);

        if (failures == 0) $display("AAGM_VERILOG_PASS");
        else $display("AAGM_VERILOG_FAIL count=%0d", failures);
        if (failures != 0) $fatal(1, "AAGM Verilog checks failed");
        $finish;
    end
endmodule
