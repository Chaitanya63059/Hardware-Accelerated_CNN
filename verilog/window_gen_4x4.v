// ============================================================================
// 4×4 Window Generator — Produces sliding 4×4 windows from pixel stream
//
// Pure Verilog — Compatible with Vivado 2025.1
//
// Architecture:
//   pix_in (row r) → [Line Buffer 0] → row r-1
//                     [Line Buffer 1] → row r-2
//                     [Line Buffer 2] → row r-3
//
//   Uses 3 line buffers (4 rows needed, current row + 3 delayed).
//   4-deep shift registers per row for column sliding.
//
// Output:
//   win_RC = window[Row][Col]:
//     Row 0 = oldest row (top), Row 3 = newest row (bottom)
//     Col 0 = left,             Col 3 = right
//
// For stride-2 operation, the controller should only capture every other
// window_valid output. Stride handling is external to this module.
// ============================================================================

`timescale 1ns / 1ps

module window_gen_4x4 #(
    parameter MAX_WIDTH = 1024
) (
    input  wire        clk,
    input  wire        rst_n,

    // Configuration
    input  wire [15:0] img_width,
    input  wire [15:0] img_height,

    // Pixel Input Stream
    input  wire [7:0]  pix_in,
    input  wire        pix_valid,
    input  wire [15:0] pix_col,
    input  wire [15:0] pix_row,

    // 4×4 Window Output
    output reg  [7:0]  win_00, win_01, win_02, win_03,   // Top row (oldest)
    output reg  [7:0]  win_10, win_11, win_12, win_13,
    output reg  [7:0]  win_20, win_21, win_22, win_23,
    output reg  [7:0]  win_30, win_31, win_32, win_33,   // Bottom row (newest)

    output reg         window_valid,
    output reg  [15:0] win_col,
    output reg  [15:0] win_row
);

    // ========================================================================
    // Line Buffers: 3 buffers to delay by 1 row each
    // lb0: row r → row r-1
    // lb1: row r-1 → row r-2
    // lb2: row r-2 → row r-3
    // ========================================================================

    wire [7:0] lb0_out, lb1_out, lb2_out;
    wire       lb0_valid, lb1_valid, lb2_valid;

    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb0 (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_width     (img_width),
        .pix_in        (pix_in),
        .pix_valid     (pix_valid),
        .pix_out       (lb0_out),
        .pix_out_valid (lb0_valid)
    );

    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_width     (img_width),
        .pix_in        (lb0_out),
        .pix_valid     (lb0_valid),
        .pix_out       (lb1_out),
        .pix_out_valid (lb1_valid)
    );

    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb2 (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_width     (img_width),
        .pix_in        (lb1_out),
        .pix_valid     (lb1_valid),
        .pix_out       (lb2_out),
        .pix_out_valid (lb2_valid)
    );

    // ========================================================================
    // 4-deep Shift Registers for each row stream
    // When lb2_valid fires, all 4 row streams are active simultaneously.
    // ========================================================================

    // Row 3 (newest/bottom): tracks pix_in
    // Row 2:                  tracks lb0_out
    // Row 1:                  tracks lb1_out
    // Row 0 (oldest/top):    tracks lb2_out

    // 4-wide shift registers (sr[0]=newest col, sr[3]=oldest col)
    reg [7:0] r0_sr0, r0_sr1, r0_sr2, r0_sr3;   // top row (oldest)
    reg [7:0] r1_sr0, r1_sr1, r1_sr2, r1_sr3;
    reg [7:0] r2_sr0, r2_sr1, r2_sr2, r2_sr3;
    reg [7:0] r3_sr0, r3_sr1, r3_sr2, r3_sr3;   // bottom row (newest)

    // Position tracking
    reg [15:0] vcol;
    reg [15:0] vrow;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r0_sr0 <= 0; r0_sr1 <= 0; r0_sr2 <= 0; r0_sr3 <= 0;
            r1_sr0 <= 0; r1_sr1 <= 0; r1_sr2 <= 0; r1_sr3 <= 0;
            r2_sr0 <= 0; r2_sr1 <= 0; r2_sr2 <= 0; r2_sr3 <= 0;
            r3_sr0 <= 0; r3_sr1 <= 0; r3_sr2 <= 0; r3_sr3 <= 0;

            win_00 <= 0; win_01 <= 0; win_02 <= 0; win_03 <= 0;
            win_10 <= 0; win_11 <= 0; win_12 <= 0; win_13 <= 0;
            win_20 <= 0; win_21 <= 0; win_22 <= 0; win_23 <= 0;
            win_30 <= 0; win_31 <= 0; win_32 <= 0; win_33 <= 0;

            window_valid <= 1'b0;
            win_col <= 0;
            win_row <= 0;
            vcol <= 0;
            vrow <= 0;
        end else begin
            window_valid <= 1'b0;   // default

            if (lb2_valid) begin
                // Shift in new pixels for each row
                r0_sr3 <= r0_sr2; r0_sr2 <= r0_sr1; r0_sr1 <= r0_sr0; r0_sr0 <= lb2_out;
                r1_sr3 <= r1_sr2; r1_sr2 <= r1_sr1; r1_sr1 <= r1_sr0; r1_sr0 <= lb1_out;
                r2_sr3 <= r2_sr2; r2_sr2 <= r2_sr1; r2_sr1 <= r2_sr0; r2_sr0 <= lb0_out;
                r3_sr3 <= r3_sr2; r3_sr2 <= r3_sr1; r3_sr1 <= r3_sr0; r3_sr0 <= pix_in;

                // Window valid after 3 columns filled (4th pixel arrives)
                if (vcol >= 3) begin
                    // Use pre-shift values (non-blocking reads current state)
                    win_00 <= r0_sr2;    win_01 <= r0_sr1;    win_02 <= r0_sr0;    win_03 <= lb2_out;
                    win_10 <= r1_sr2;    win_11 <= r1_sr1;    win_12 <= r1_sr0;    win_13 <= lb1_out;
                    win_20 <= r2_sr2;    win_21 <= r2_sr1;    win_22 <= r2_sr0;    win_23 <= lb0_out;
                    win_30 <= r3_sr2;    win_31 <= r3_sr1;    win_32 <= r3_sr0;    win_33 <= pix_in;

                    window_valid <= 1'b1;
                    win_col <= vcol - 3;
                    win_row <= vrow;
                end

                // Advance counters
                if (vcol == img_width - 1) begin
                    vcol <= 0;
                    vrow <= vrow + 1;
                end else begin
                    vcol <= vcol + 1;
                end
            end
        end
    end

endmodule
