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
        #20;

        // 1. Configure Budget to 8 steps
        bus_write(5'h14, 32'd8);

        // 2. Start Inference
        bus_write(5'h00, 32'h01); // CTRL_REG start

        // 3. Post Step 1: g_1 = 0.6058 (19851 in Q1.15), shadow s_2 = 0.5466 (17911)
        bus_write(5'h0C, 32'd17911); // shadow
        bus_write(5'h08, 32'd19851); // gate
        #10;
        if (irq_halt !== 1'b0) begin
            $display("FAIL: Halt triggered at step 1 (min 2 steps required)");
            failures = failures + 1;
        end

        // 4. Post Step 2: g_2 = 0.3763 (12330 in Q1.15), shadow s_3 = 0.2800 (9175 < 11370) -> PREARM!
        bus_write(5'h0C, 32'd9175); // shadow < tau_lo
        #10;
        if (speculative_prearm !== 1'b1) begin
            $display("FAIL: Speculative prearm not asserted for shadow < tau_lo");
            failures = failures + 1;
        end

        bus_write(5'h08, 32'd12330); // gate -> cumulative mass >= 29491
        #10;
        if (irq_halt !== 1'b1) begin
            $display("FAIL: Halt not triggered at step 2 when threshold met");
            failures = failures + 1;
        end

        // 5. Test Asynchronous Mid-Inference Abort
        bus_write(5'h00, 32'h01); // restart
        #10;
        bus_write(5'h00, 32'h04); // trigger abort (bit 2)
        #10;
        if (irq_abort !== 1'b1) begin
            $display("FAIL: Async abort interrupt not asserted");
            failures = failures + 1;
        end

        #20;
        if (failures == 0) $display("AAGM_COPROCESSOR_VERILOG_PASS");
        else $display("AAGM_COPROCESSOR_VERILOG_FAIL count=%0d", failures);
        $finish;
    end

endmodule
