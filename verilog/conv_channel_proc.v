// ============================================================================
// Conv Channel Processor — One processing lane (engine + BRAMs)
//
// Pure Verilog — Compatible with Vivado 2025.1
//
// Contains one complete convolution lane:
//   - Weight BRAM (loaded from PS)
//   - Conv Engine (MAC)
//   - Accumulator BRAM (read-modify-write)
//   - Output BRAM (bias + ReLU results)
//
// Multiple instances share the same window input but have different weights.
// All control signals (addresses, enables) are broadcast from the parent FSM.
// ============================================================================

`timescale 1ns / 1ps

module conv_channel_proc #(
    parameter WBRAM_DEPTH = 4096,
    parameter ABRAM_DEPTH = 4096
) (
    input  wire        clk,
    input  wire        rst_n,

    // ========== Weight BRAM Write (from PS via AXI-Lite) ==========
    input  wire [11:0] w_wr_addr,
    input  wire [7:0]  w_wr_data,
    input  wire        w_wr_en,        // Only this lane's enable

    // ========== Kernel Loading (from FSM) ==========
    input  wire [11:0] w_rd_addr,      // Shared read address
    input  wire [3:0]  kern_load_idx,  // Engine register index
    input  wire        kern_load_en,   // Load weight into engine

    // ========== Conv Engine Config ==========
    input  wire [1:0]  kernel_mode,

    // ========== Shared Window Input ==========
    input  wire [7:0]  win_00, win_01, win_02, win_03,
    input  wire [7:0]  win_10, win_11, win_12, win_13,
    input  wire [7:0]  win_20, win_21, win_22, win_23,
    input  wire [7:0]  win_30, win_31, win_32, win_33,
    input  wire        window_valid,

    // ========== Accumulator Control (from FSM) ==========
    input  wire [11:0] acc_rd_addr,    // Read address
    input  wire        acc_rmw_trigger,// Start RMW cycle
    input  wire [11:0] acc_rmw_addr,   // Write-back address
    input  wire        acc_first_ch,   // First channel: overwrite, not accumulate

    // ========== Bias + ReLU (from FSM) ==========
    input  wire signed [15:0] bias_val, // This lane's bias
    input  wire [4:0]  out_shift,       // Arithmetic right shift before clamp
        input  wire [11:0] out_wr_addr,     // Shared write address for output
    input  wire        out_wr_en,       // Shared write enable for output

    // ========== Output Read (from FSM) ==========
    input  wire [11:0] out_rd_addr,
    output reg  [7:0]  out_rd_data
);

    // ========================================================================
    // Weight BRAM
    // ========================================================================

    (* ram_style = "block" *)
    reg [7:0] weight_bram [0:WBRAM_DEPTH-1];
    reg [7:0] w_rd_data;

    always @(posedge clk) begin
        if (w_wr_en)
            weight_bram[w_wr_addr] <= w_wr_data;
    end

    always @(posedge clk) begin
        w_rd_data <= weight_bram[w_rd_addr];
    end

    // ========================================================================
    // Conv Engine Instance
    // ========================================================================

    wire signed [23:0] eng_conv_sum;
    wire               eng_sum_valid;

    conv_engine eng_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .kernel_mode  (kernel_mode),
        .w_load_data  (w_rd_data),
        .w_load_idx   (kern_load_idx),
        .w_load_en    (kern_load_en),
        .win_00(win_00), .win_01(win_01), .win_02(win_02), .win_03(win_03),
        .win_10(win_10), .win_11(win_11), .win_12(win_12), .win_13(win_13),
        .win_20(win_20), .win_21(win_21), .win_22(win_22), .win_23(win_23),
        .win_30(win_30), .win_31(win_31), .win_32(win_32), .win_33(win_33),
        .window_valid (window_valid),
        .conv_sum     (eng_conv_sum),
        .sum_valid    (eng_sum_valid)
    );

    // ========================================================================
    // Accumulator BRAM — Read-Modify-Write
    // ========================================================================

    (* ram_style = "block" *)
    reg signed [31:0] acc_bram [0:ABRAM_DEPTH-1];
    reg signed [31:0] acc_rd_data;

    // Read port
    always @(posedge clk) begin
        acc_rd_data <= acc_bram[acc_rd_addr];
    end

    // RMW pipeline: trigger → read (1 cycle) → compute+write (1 cycle)
    reg        rmw_pending;
    reg [11:0] rmw_addr;
    reg        rmw_first;
    reg signed [23:0] rmw_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rmw_pending <= 1'b0;
        end else begin
            if (acc_rmw_trigger && eng_sum_valid) begin
                rmw_pending <= 1'b1;
                rmw_addr    <= acc_rmw_addr;
                rmw_first   <= acc_first_ch;
                rmw_sum     <= eng_conv_sum;
            end else if (rmw_pending) begin
                rmw_pending <= 1'b0;
            end
        end
    end

    // BRAM Write Port (must not have asynchronous reset for BRAM inference)
    always @(posedge clk) begin
        if (rmw_pending) begin
            if (rmw_first)
                acc_bram[rmw_addr] <= {{8{rmw_sum[23]}}, rmw_sum};
            else
                acc_bram[rmw_addr] <= acc_rd_data + {{8{rmw_sum[23]}}, rmw_sum};
        end
    end

    // ========================================================================
    // Output BRAM + Bias/ReLU
    // ========================================================================

    (* ram_style = "block" *)
    reg [7:0] out_bram [0:ABRAM_DEPTH-1];

    // Bias + ReLU: combinational
    wire signed [31:0] biased;
        assign biased = acc_rd_data + {{16{bias_val[15]}}, bias_val};
    
    wire signed [31:0] shifted;
    assign shifted = biased >>> out_shift;

    wire [7:0] relu_out;
    assign relu_out = (shifted[31])      ? 8'h00 :   // Negative → 0
                      (|shifted[30:7])    ? 8'h7F :   // > 127 → saturate
                      shifted[7:0];

    // Bias+ReLU write: combinational relu_out + synchronous BRAM write
    always @(posedge clk) begin
        if (out_wr_en) begin
            out_bram[out_wr_addr] <= relu_out;
        end
    end

    // Output read
    always @(posedge clk) begin
        out_rd_data <= out_bram[out_rd_addr];
    end

endmodule
