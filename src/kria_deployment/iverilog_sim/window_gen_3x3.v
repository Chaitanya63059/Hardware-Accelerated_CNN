// ============================================================================
// 3x3 Window Generator - Produces sliding 3x3 windows from pixel stream
//
// Pure Verilog (No SystemVerilog)
// Compatible with Vivado 2025.1
//
// Architecture:
//   pix_in (row r) ──→ [Line Buffer 0] ──→ row_1 (row r-1)
//                       [Line Buffer 1] ──→ row_2 (row r-2)
//
//   NOTE: Line buffers have BRAM read latency — pix_out_valid fires
//   1 cycle after reading. This module stores each row in a 3-deep
//   shift register, aligning all three rows correctly.
//
// Output:
//   win_RC = window[Row][Col]:
//     Row 0 = oldest row (top), Row 2 = newest row (bottom)
//     Col 0 = left,             Col 2 = right
//
// Parameters:
//   MAX_WIDTH: Maximum supported image width (passed to line buffers)
// ============================================================================

`timescale 1ns / 1ps

module window_gen_3x3 #(
    parameter MAX_WIDTH = 1024
) (
    input  wire        clk,
    input  wire        rst_n,

    // Configuration
    input  wire [15:0] img_width,
    input  wire [15:0] img_height,

    // Pixel Input Stream (from pixel_streamer)
    input  wire [7:0]  pix_in,
    input  wire        pix_valid,
    input  wire [15:0] pix_col,
    input  wire [15:0] pix_row,

    // 3x3 Window Output
    output reg  [7:0]  win_00, win_01, win_02,   // Top row    (oldest)
    output reg  [7:0]  win_10, win_11, win_12,   // Middle row
    output reg  [7:0]  win_20, win_21, win_22,   // Bottom row (newest)

    output reg         window_valid,
    output reg  [15:0] win_col,
    output reg  [15:0] win_row
);

    // ========================================================================
    // Line Buffers: delay pixel stream by 1 row each
    // ========================================================================

    wire [7:0] lb0_out;
    wire       lb0_valid;
    wire [7:0] lb1_out;
    wire       lb1_valid;

    // lb0: delays pix_in by 1 row → row r-1 data
    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb0 (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_width     (img_width),
        .pix_in        (pix_in),
        .pix_valid     (pix_valid),
        .pix_out       (lb0_out),
        .pix_out_valid (lb0_valid)
    );

    // lb1: delays lb0_out by 1 more row → row r-2 data
    line_buffer #(.MAX_WIDTH(MAX_WIDTH)) lb1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .img_width     (img_width),
        .pix_in        (lb0_out),
        .pix_valid     (lb0_valid),
        .pix_out       (lb1_out),
        .pix_out_valid (lb1_valid)
    );

    // ========================================================================
    // 3-deep Shift Registers for each row stream
    // When lb1_valid fires, all 3 row streams are active simultaneously.
    // We shift each row's last 3 pixels to form the 3-column window.
    // ========================================================================

    // Row 0 (newest, bottom of window): tracks pix_in (delayed 2 rows via lb timing)
    // Row 1 (middle):                   tracks lb0_out
    // Row 2 (oldest, top of window):    tracks lb1_out

    // 3-wide shift registers (sr[0]=newest col, sr[2]=oldest col)
    reg [7:0] r0_sr [0:2];   // bottom row pixels
    reg [7:0] r1_sr [0:2];   // middle row pixels
    reg [7:0] r2_sr [0:2];   // top row pixels

    // Position tracking (counts lb1's output position)
    reg [15:0] vcol;         // column within lb1 output
    reg [15:0] vrow;         // row within lb1 output

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r0_sr[0] <= 0; r0_sr[1] <= 0; r0_sr[2] <= 0;
            r1_sr[0] <= 0; r1_sr[1] <= 0; r1_sr[2] <= 0;
            r2_sr[0] <= 0; r2_sr[1] <= 0; r2_sr[2] <= 0;
            win_00 <= 0; win_01 <= 0; win_02 <= 0;
            win_10 <= 0; win_11 <= 0; win_12 <= 0;
            win_20 <= 0; win_21 <= 0; win_22 <= 0;
            window_valid <= 1'b0;
            win_col <= 0;
            win_row <= 0;
            vcol <= 0;
            vrow <= 0;
        end else begin
            window_valid <= 1'b0;  // default

            if (lb1_valid) begin
                // Shift in new pixel for each row (oldest elem shifts out)
                r2_sr[2] <= r2_sr[1]; r2_sr[1] <= r2_sr[0]; r2_sr[0] <= lb1_out;
                r1_sr[2] <= r1_sr[1]; r1_sr[1] <= r1_sr[0]; r1_sr[0] <= lb0_out;
                r0_sr[2] <= r0_sr[1]; r0_sr[1] <= r0_sr[0]; r0_sr[0] <= pix_in;

                // Window is valid after 2 complete column shifts (3 cols filled)
                // Use the just-shifted values — they'll be ready next cycle
                if (vcol >= 2) begin
                    // Capture window on THIS cycle (uses pre-shift values below)
                    // Top row (oldest, r2): sr[2]=left, sr[1]=mid, sr[0]=right (before shift=new)
                    // After non-blocking: sr[2]←sr[1], sr[1]←sr[0], sr[0]←lb1_out
                    // So we need to use CURRENT values (before assignment takes effect)
                    // In an always block, non-blocking reads use PRE-clock values
                    win_00 <= r2_sr[1];   // will become sr[2] after shift -- pre-shift sr[1]
                    win_01 <= r2_sr[0];   // will become sr[1] after shift
                    win_02 <= lb1_out;    // will become sr[0] after shift (newest)

                    win_10 <= r1_sr[1];
                    win_11 <= r1_sr[0];
                    win_12 <= lb0_out;

                    win_20 <= r0_sr[1];
                    win_21 <= r0_sr[0];
                    win_22 <= pix_in;

                    window_valid <= 1'b1;
                    win_col <= vcol - 2;
                    win_row <= vrow;
                end

                // Advance column/row counters
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
