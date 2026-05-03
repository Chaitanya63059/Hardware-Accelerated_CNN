// ============================================================================
// Line Buffer - Delays a pixel stream by one full image row
//
// Pure Verilog (No SystemVerilog)
// Compatible with Vivado 2025.1
//
// Purpose:
//   In a sliding-window convolution, the window spans K rows. To form
//   the window from a serial pixel stream, we need K-1 line buffers,
//   each delaying the stream by exactly one row (W pixels).
//
// Implementation:
//   Uses a circular buffer (BRAM-inferred) with separate read/write
//   pointers. Depth = image width → 1-row delay.
//
//   For widths <= 1024: infers BRAM (2 × 18Kb BRAM per buffer)
//   For widths <= 64:   synthesizes to SRL16/SRL32 shift registers
//
// Data Flow:
//   pix_in (current row) ──→ [Line Buffer, W deep] ──→ pix_out (previous row)
//   Both pix_in and pix_out are valid on the same cycle when valid=1
//
// Latency: W clock cycles (one full row)
//
// Parameters:
//   MAX_WIDTH: Maximum supported image width (determines BRAM depth)
//              Actual width set at runtime via img_width input
// ============================================================================

`timescale 1ns / 1ps

module line_buffer #(
    parameter MAX_WIDTH = 1024    // Max image width (BRAM depth)
) (
    input  wire        clk,
    input  wire        rst_n,

    // Configuration
    input  wire [15:0] img_width,     // Actual image width (runtime)

    // Pixel Input
    input  wire [7:0]  pix_in,        // Input pixel
    input  wire        pix_valid,     // Input valid

    // Pixel Output (delayed by img_width cycles)
    output reg  [7:0]  pix_out,       // Delayed pixel (1 row behind)
    output reg         pix_out_valid  // Output valid
);

    // ========================================================================
    // Circular Buffer Memory (infers BRAM in Vivado)
    // ========================================================================

    (* ram_style = "block" *)
    reg [7:0] buffer_mem [0:MAX_WIDTH-1];

    // Pointers
    reg [15:0] wr_ptr;
    reg [15:0] fill_count;    // How many valid pixels stored
    reg        buffer_full;   // Set when fill_count >= img_width

    // ========================================================================
    // Write and Read Logic
    // ========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr        <= 16'h0;
            fill_count    <= 16'h0;
            buffer_full   <= 1'b0;
            pix_out       <= 8'h00;
            pix_out_valid <= 1'b0;
        end else begin
            pix_out_valid <= 1'b0;  // default

            if (pix_valid) begin
                // Read BEFORE write (old data at write pointer = delayed output)
                if (buffer_full) begin
                    pix_out       <= buffer_mem[wr_ptr];
                    pix_out_valid <= 1'b1;
                end

                // Write new pixel
                buffer_mem[wr_ptr] <= pix_in;

                // Advance write pointer (circular)
                if (wr_ptr == img_width - 1)
                    wr_ptr <= 16'h0;
                else
                    wr_ptr <= wr_ptr + 1;

                // Track fill level
                if (!buffer_full) begin
                    if (fill_count + 1 >= img_width)
                        buffer_full <= 1'b1;
                    fill_count <= fill_count + 1;
                end
            end
        end
    end

endmodule
