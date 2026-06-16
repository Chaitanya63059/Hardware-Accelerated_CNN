// ============================================================================
// Parameterized Convolution MAC Engine
//
// Pure Verilog — Compatible with Vivado 2025.1
//
// Supports 1×1, 3×3, and 4×4 kernels via kernel_mode input.
// Weights stored in a 4×4 grid (16 registers). For smaller kernels,
// unused positions must be loaded with 0 by the controller.
//
// Compute: result = Σ win[r][c] * weight[r][c]  (up to 16 MACs)
//
// All 16 products are always computed; zero weights/windows ensure
// unused products contribute nothing. Single-cycle combinational MAC.
//
// INT8 signed weights × INT8 unsigned activations → 24-bit signed result.
// ============================================================================

`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module conv_engine (
    input  wire        clk,
    input  wire        rst_n,

    // ========== Configuration ==========
    input  wire [1:0]  kernel_mode,     // 2'b00=1×1, 2'b01=3×3, 2'b10=4×4

    // ========== Weight Loading ==========
    // Index 0-15, mapped as row*4+col in the 4×4 grid
    input  wire [7:0]  w_load_data,     // INT8 signed weight
    input  wire [3:0]  w_load_idx,      // 0-15
    input  wire        w_load_en,

    // ========== Window Input (4×4 max) ==========
    // Row 0 (top)
    input  wire [7:0]  win_00, win_01, win_02, win_03,
    // Row 1
    input  wire [7:0]  win_10, win_11, win_12, win_13,
    // Row 2
    input  wire [7:0]  win_20, win_21, win_22, win_23,
    // Row 3 (bottom, used only in 4×4 mode)
    input  wire [7:0]  win_30, win_31, win_32, win_33,

    input  wire        window_valid,

    // ========== Output ==========
    output reg  signed [23:0] conv_sum,     // MAC result (signed)
    output reg                sum_valid     // Result valid strobe
);

    // ========================================================================
    // Weight Storage — 4×4 grid of INT8 signed registers
    // ========================================================================

    reg signed [7:0] w00, w01, w02, w03;
    reg signed [7:0] w10, w11, w12, w13;
    reg signed [7:0] w20, w21, w22, w23;
    reg signed [7:0] w30, w31, w32, w33;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w00 <= 8'sd0; w01 <= 8'sd0; w02 <= 8'sd0; w03 <= 8'sd0;
            w10 <= 8'sd0; w11 <= 8'sd0; w12 <= 8'sd0; w13 <= 8'sd0;
            w20 <= 8'sd0; w21 <= 8'sd0; w22 <= 8'sd0; w23 <= 8'sd0;
            w30 <= 8'sd0; w31 <= 8'sd0; w32 <= 8'sd0; w33 <= 8'sd0;
        end else if (w_load_en) begin
            case (w_load_idx)
                4'd0:  w00 <= w_load_data;
                4'd1:  w01 <= w_load_data;
                4'd2:  w02 <= w_load_data;
                4'd3:  w03 <= w_load_data;
                4'd4:  w10 <= w_load_data;
                4'd5:  w11 <= w_load_data;
                4'd6:  w12 <= w_load_data;
                4'd7:  w13 <= w_load_data;
                4'd8:  w20 <= w_load_data;
                4'd9:  w21 <= w_load_data;
                4'd10: w22 <= w_load_data;
                4'd11: w23 <= w_load_data;
                4'd12: w30 <= w_load_data;
                4'd13: w31 <= w_load_data;
                4'd14: w32 <= w_load_data;
                4'd15: w33 <= w_load_data;
                default: ;
            endcase
        end
    end

    // ========================================================================
    // Combinational MAC — 16 multiplies + adder tree
    // ========================================================================
    // Each product: signed 8-bit weight × unsigned 8-bit activation
    // Product width: 16 bits signed
    // Sum of 16 products: needs 20 bits, we use 24 for safety

    // Row 0 products
    (* use_dsp = "yes" *) wire signed [15:0] p00, p01, p02, p03;
    assign p00 = w00 * $signed({1'b0, win_00});
    assign p01 = w01 * $signed({1'b0, win_01});
    assign p02 = w02 * $signed({1'b0, win_02});
    assign p03 = w03 * $signed({1'b0, win_03});

    // Row 1 products
    (* use_dsp = "yes" *) wire signed [15:0] p10, p11, p12, p13;
    assign p10 = w10 * $signed({1'b0, win_10});
    assign p11 = w11 * $signed({1'b0, win_11});
    assign p12 = w12 * $signed({1'b0, win_12});
    assign p13 = w13 * $signed({1'b0, win_13});

    // Row 2 products
    (* use_dsp = "yes" *) wire signed [15:0] p20, p21, p22, p23;
    assign p20 = w20 * $signed({1'b0, win_20});
    assign p21 = w21 * $signed({1'b0, win_21});
    assign p22 = w22 * $signed({1'b0, win_22});
    assign p23 = w23 * $signed({1'b0, win_23});

    // Row 3 products
    (* use_dsp = "yes" *) wire signed [15:0] p30, p31, p32, p33;
    assign p30 = w30 * $signed({1'b0, win_30});
    assign p31 = w31 * $signed({1'b0, win_31});
    assign p32 = w32 * $signed({1'b0, win_32});
    assign p33 = w33 * $signed({1'b0, win_33});

    // Adder tree — balanced for timing
    // Level 1: pairs
    wire signed [16:0] s00_01, s02_03, s10_11, s12_13;
    wire signed [16:0] s20_21, s22_23, s30_31, s32_33;
    assign s00_01 = {p00[15], p00} + {p01[15], p01};
    assign s02_03 = {p02[15], p02} + {p03[15], p03};
    assign s10_11 = {p10[15], p10} + {p11[15], p11};
    assign s12_13 = {p12[15], p12} + {p13[15], p13};
    assign s20_21 = {p20[15], p20} + {p21[15], p21};
    assign s22_23 = {p22[15], p22} + {p23[15], p23};
    assign s30_31 = {p30[15], p30} + {p31[15], p31};
    assign s32_33 = {p32[15], p32} + {p33[15], p33};

    // Level 2: quads
    wire signed [17:0] q_r0, q_r1, q_r2, q_r3;
    assign q_r0 = {s00_01[16], s00_01} + {s02_03[16], s02_03};
    assign q_r1 = {s10_11[16], s10_11} + {s12_13[16], s12_13};
    assign q_r2 = {s20_21[16], s20_21} + {s22_23[16], s22_23};
    assign q_r3 = {s30_31[16], s30_31} + {s32_33[16], s32_33};

    // Level 3: row pairs
    wire signed [18:0] h_01, h_23;
    assign h_01 = {q_r0[17], q_r0} + {q_r1[17], q_r1};
    assign h_23 = {q_r2[17], q_r2} + {q_r3[17], q_r3};

    // Level 4: final sum
    wire signed [19:0] mac_total;
    assign mac_total = {h_01[18], h_01} + {h_23[18], h_23};

    // Sign-extend to 24 bits
    wire signed [23:0] mac_result;
    assign mac_result = {{4{mac_total[19]}}, mac_total};

    // ========================================================================
    // Registered Output (1-cycle latency)
    // ========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_sum  <= 24'sd0;
            sum_valid <= 1'b0;
        end else begin
            sum_valid <= window_valid;
            if (window_valid) begin
                conv_sum <= mac_result;
            end
        end
    end

endmodule
