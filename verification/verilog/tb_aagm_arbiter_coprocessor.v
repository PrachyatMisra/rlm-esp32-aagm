// tb_aagm_arbiter_coprocessor.v
// Testbench for the AAGM Hardware Arbiter Coprocessor

`timescale 1ns / 1ps

module tb_aagm_arbiter_coprocessor;
    reg         clk;
    reg         rst_n;
    reg         p_sel;
    reg         p_enable;
    reg         p_write;
    reg  [4:0]  p_addr;
    reg  [31:0] p_wdata;
    wire [31:0] p_rdata;
    wire        p_ready;
    wire        irq_halt;
    wire        irq_abort;
    wire        speculative_prearm;

    integer failures;

    aagm_arbiter_coprocessor dut (
        .clk(clk),
        .rst_n(rst_n),
        .p_sel(p_sel),
        .p_enable(p_enable),
        .p_write(p_write),
        .p_addr(p_addr),
        .p_wdata(p_wdata),
        .p_rdata(p_rdata),
        .p_ready(p_ready),
        .irq_halt(irq_halt),
        .irq_abort(irq_abort),
        .speculative_prearm(speculative_prearm)
    );

    always #5 clk = ~clk; // 100 MHz clock

    task bus_write;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk);
            p_sel    = 1'b1;
            p_write  = 1'b1;
            p_addr   = addr;
            p_wdata  = data;
            @(posedge clk);
            p_enable = 1'b1;
            @(posedge clk);
            p_sel    = 1'b0;
            p_enable = 1'b0;
            p_write  = 1'b0;
        end
    endtask

    initial begin
        $dumpfile("aagm_coprocessor.vcd");
        $dumpvars(0, tb_aagm_arbiter_coprocessor);
        failures = 0;
        clk = 0;
        rst_n = 0;
        p_sel = 0;
        p_enable = 0;
        p_write = 0;
        p_addr = 0;
        p_wdata = 0;

        #20;
        rst_n = 1;
        $display("[TIME %0tps] RST_N deasserted (active-high operation).", $time);
        #20;

        // 1. Configure Budget to 8 steps
        $display("[TIME %0tps] APB WRITE ADDR_BUDGET (0x14) <= 8", $time);
        bus_write(5'h14, 32'd8);

        // 2. Start Inference
        $display("[TIME %0tps] APB WRITE ADDR_CTRL (0x00) <= 0x01 [START]", $time);
        bus_write(5'h00, 32'h01); // CTRL_REG start

        // 3. Post Step 1: g_1 = 0.6058 (19851 in Q1.15), shadow s_2 = 0.5466 (17911)
        $display("[TIME %0tps] APB WRITE ADDR_SHADOW_IN (0x0C) <= 17911 (s_2 = 0.5466)", $time);
        bus_write(5'h0C, 32'd17911); // shadow
        $display("[TIME %0tps] APB WRITE ADDR_GATE_IN (0x08) <= 19851 (g_1 = 0.6058)", $time);
        bus_write(5'h08, 32'd19851); // gate
        #10;
        $display("[TIME %0tps] CHECK Step 1: cum_mass=%0d/29491, steps=%0d, irq_halt=%b, prearm=%b",
                 $time, dut.mass_acc_reg, dut.steps_reg, irq_halt, speculative_prearm);
        if (irq_halt !== 1'b0) begin
            $display("FAIL: Halt triggered at step 1 (min 2 steps required)");
            failures = failures + 1;
        end

        // 4. Post Step 2: g_2 = 0.3763 (12330 in Q1.15), shadow s_3 = 0.2800 (9175 < 11370) -> PREARM!
        $display("[TIME %0tps] APB WRITE ADDR_SHADOW_IN (0x0C) <= 9175 (s_3 = 0.2800 < tau_lo 0.347)", $time);
        bus_write(5'h0C, 32'd9175); // shadow < tau_lo
        #10;
        $display("[TIME %0tps] CHECK Speculative Prearm: prearm=%b (EXPECT 1)", $time, speculative_prearm);
        if (speculative_prearm !== 1'b1) begin
            $display("FAIL: Speculative prearm not asserted for shadow < tau_lo");
            failures = failures + 1;
        end

        $display("[TIME %0tps] APB WRITE ADDR_GATE_IN (0x08) <= 12330 (g_2 = 0.3763)", $time);
        bus_write(5'h08, 32'd12330); // gate -> cumulative mass >= 29491
        #10;
        $display("[TIME %0tps] CHECK Step 2: cum_mass=%0d/29491, steps=%0d, irq_halt=%b (EXPECT 1)",
                 $time, dut.mass_acc_reg, dut.steps_reg, irq_halt);
        if (irq_halt !== 1'b1) begin
            $display("FAIL: Halt not triggered at step 2 when threshold met");
            failures = failures + 1;
        end

        // 5. Test Asynchronous Mid-Inference Abort
        $display("[TIME %0tps] TEST Mid-Inference Asynchronous Abort Interrupt...", $time);
        bus_write(5'h00, 32'h01); // restart
        #10;
        bus_write(5'h00, 32'h04); // trigger abort (bit 2)
        #10;
        $display("[TIME %0tps] CHECK Async Abort: irq_abort=%b (EXPECT 1)", $time, irq_abort);
        if (irq_abort !== 1'b1) begin
            $display("FAIL: Async abort interrupt not asserted");
            failures = failures + 1;
        end

        #20;
        if (failures == 0) $display(">>> AAGM_COPROCESSOR_VERILOG_PASS (All hardware checks verified)");
        else $display(">>> AAGM_COPROCESSOR_VERILOG_FAIL count=%0d", failures);
        $finish;
    end

endmodule
