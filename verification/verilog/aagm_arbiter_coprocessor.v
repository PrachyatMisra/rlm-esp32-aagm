// aagm_arbiter_coprocessor.v
// Hardware Coprocessor for Asynchronous Adaptive Gating Mechanism (AAGM)
// Implements memory-mapped register interface, hardware mass accumulator,
// calibrated shadow forecaster comparator, dynamic recursion budget counter,
// and single-cycle asynchronous abort interrupt.

`timescale 1ns / 1ps

module aagm_arbiter_coprocessor (
    input  wire        clk,
    input  wire        rst_n,

    // Memory-Mapped Register Bus Interface (APB-compatible)
    input  wire        p_sel,
    input  wire        p_enable,
    input  wire        p_write,
    input  wire [4:0]  p_addr,
    input  wire [31:0] p_wdata,
    output reg  [31:0] p_rdata,
    output wire        p_ready,

    // Hardware Interrupt & Control Flags
    output wire        irq_halt,
    output wire        irq_abort,
    output wire        speculative_prearm
);

    assign p_ready = 1'b1;

    // Register Map Offsets
    localparam ADDR_CTRL        = 5'h00; // [0]=start, [1]=soft_reset, [2]=async_abort, [3]=spec_en
    localparam ADDR_STATUS      = 5'h04; // [0]=busy, [1]=halted, [2]=aborted, [3]=prearmed
    localparam ADDR_GATE_IN     = 5'h08; // [15:0]=g_k in Q1.15
    localparam ADDR_SHADOW_IN   = 5'h0C; // [15:0]=s_{k+1} in Q1.15
    localparam ADDR_MASS_ACC    = 5'h10; // [15:0]=cumulative halting mass in Q1.15
    localparam ADDR_BUDGET      = 5'h14; // [3:0]=max recursion budget (4, 6, 8)
    localparam ADDR_STEPS       = 5'h18; // [3:0]=current step count
    localparam ADDR_TAU_LO      = 5'h1C; // [15:0]=shadow threshold (0.347 in Q1.15 = 11370)

    // Golden Constants
    localparam [15:0] THRESHOLD_Q15 = 16'd29491; // 0.900 in Q1.15
    localparam [15:0] ONE_Q15       = 16'd32768; // 1.000 in Q1.15
    localparam [15:0] DEFAULT_TAU   = 16'd11370; // 0.347 in Q1.15

    // Internal Registers
    reg [3:0]  ctrl_reg;
    reg        busy_flag;
    reg        halt_flag;
    reg        abort_flag;
    reg        prearm_flag;
    reg [15:0] gate_reg;
    reg [15:0] shadow_reg;
    reg [15:0] mass_acc_reg;
    reg [3:0]  budget_reg;
    reg [3:0]  steps_reg;
    reg [15:0] tau_lo_reg;

    // Combinational Outputs
    assign irq_halt = halt_flag;
    assign irq_abort = abort_flag;
    assign speculative_prearm = prearm_flag;

    // Bus Read Operation
    always @(*) begin
        case (p_addr)
            ADDR_CTRL:      p_rdata = {28'b0, ctrl_reg};
            ADDR_STATUS:    p_rdata = {28'b0, prearm_flag, abort_flag, halt_flag, busy_flag};
            ADDR_GATE_IN:   p_rdata = {16'b0, gate_reg};
            ADDR_SHADOW_IN: p_rdata = {16'b0, shadow_reg};
            ADDR_MASS_ACC:  p_rdata = {16'b0, mass_acc_reg};
            ADDR_BUDGET:    p_rdata = {28'b0, budget_reg};
            ADDR_STEPS:     p_rdata = {28'b0, steps_reg};
            ADDR_TAU_LO:    p_rdata = {16'b0, tau_lo_reg};
            default:        p_rdata = 32'b0;
        endcase
    end

    // Bus Write & Sequential Coprocessor Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg     <= 4'b0;
            busy_flag    <= 1'b0;
            halt_flag    <= 1'b0;
            abort_flag   <= 1'b0;
            prearm_flag  <= 1'b0;
            gate_reg     <= 16'b0;
            shadow_reg   <= 16'b0;
            mass_acc_reg <= 16'b0;
            budget_reg   <= 4'd8; // Default PERF budget
            steps_reg    <= 4'b0;
            tau_lo_reg   <= DEFAULT_TAU;
        end else begin
            // Soft reset check
            if (ctrl_reg[1]) begin
                busy_flag    <= 1'b0;
                halt_flag    <= 1'b0;
                abort_flag   <= 1'b0;
                prearm_flag  <= 1'b0;
                mass_acc_reg <= 16'b0;
                steps_reg    <= 4'b0;
                ctrl_reg[1]  <= 1'b0;
            end

            // Asynchronous Abort Trigger
            if (ctrl_reg[2]) begin
                abort_flag <= 1'b1;
                busy_flag  <= 1'b0;
            end

            // Bus Write Processing
            if (p_sel && p_enable && p_write) begin
                case (p_addr)
                    ADDR_CTRL: begin
                        ctrl_reg <= p_wdata[3:0];
                        if (p_wdata[0]) begin // Start new inference
                            busy_flag    <= 1'b1;
                            halt_flag    <= 1'b0;
                            abort_flag   <= 1'b0;
                            prearm_flag  <= 1'b0;
                            mass_acc_reg <= 16'b0;
                            steps_reg    <= 4'b0;
                        end
                    end
                    ADDR_GATE_IN: begin
                        gate_reg <= p_wdata[15:0];
                        if (busy_flag && !halt_flag && !abort_flag) begin
                            // Accumulate mass: mass += (1.0 - g_k)
                            // In Q1.15: delta = 32768 - gate
                            steps_reg <= steps_reg + 1'b1;
                            if (p_wdata[15:0] <= ONE_Q15) begin
                                mass_acc_reg <= mass_acc_reg + (ONE_Q15 - p_wdata[15:0]);
                            end

                            // Evaluate ACT halting condition:
                            // k >= 2 and cumulative mass >= THRESHOLD_Q15 and not budget exhausted
                            if ((steps_reg + 1'b1 >= 4'd2) &&
                                (mass_acc_reg + (ONE_Q15 - p_wdata[15:0]) >= THRESHOLD_Q15)) begin
                                halt_flag <= 1'b1;
                                busy_flag <= 1'b0;
                            end else if (steps_reg + 1'b1 >= budget_reg) begin
                                // Budget exhausted override
                                halt_flag <= 1'b1;
                                busy_flag <= 1'b0;
                            end
                        end
                    end
                    ADDR_SHADOW_IN: begin
                        shadow_reg <= p_wdata[15:0];
                        // Evaluate speculative shadow forecaster: s_{k+1} < tau_lo
                        if (p_wdata[15:0] < tau_lo_reg) begin
                            prearm_flag <= 1'b1;
                        end else begin
                            prearm_flag <= 1'b0;
                        end
                    end
                    ADDR_BUDGET: begin
                        budget_reg <= p_wdata[3:0];
                    end
                    ADDR_TAU_LO: begin
                        tau_lo_reg <= p_wdata[15:0];
                    end
                endcase
            end
        end
    end

endmodule
