// ============================================================================
// CNN Compute Unit — 16-Parallel Output Channel Processing
//
// Pure Verilog — Compatible with Vivado 2025.1
//
// Processes 16 output channels SIMULTANEOUSLY using 16 parallel conv_channel_proc
// lanes. Each lane has its own weight BRAM, conv engine, accumulator, and
// output buffer — but they all share the SAME window generator output.
//
// For each PS-triggered run:
//   1. PS pre-loads 16 sets of weights (one per lane) + 16 bias values
//   2. PS DMAs the input feature map (C_in × H × W)
//   3. All 16 lanes process simultaneously, accumulating across C_in channels
//   4. All 16 lanes apply bias + ReLU in parallel
//   5. Module streams 16 × H_out × W_out result bytes via AXI-Stream
//
// PS loops: for oc_group in range(0, C_out, 16)
//   → loads weights for oc_group..oc_group+15
//   → starts, DMAs input, receives 16 output channels
//
// Performance: 16× throughput vs single-engine design
// Resources: ~256 DSP48E2 (~20% of KV260), ~384 KB BRAM
// ============================================================================

`timescale 1ns / 1ps

module cnn_compute_unit #(
    parameter N_PARALLEL  = 16,     // Number of parallel engines
    parameter MAX_WIDTH   = 128,    // Max input width
    parameter WBRAM_DEPTH = 4096,   // Weight BRAM depth per lane
    parameter ABRAM_DEPTH = 4096    // Accumulator depth per lane
) (
    input  wire        clk,
    input  wire        rst_n,

    // ========== Configuration (from AXI-Lite) ==========
    input  wire [1:0]  kernel_mode,
    input  wire [1:0]  stride,
    input  wire [15:0] img_width,
    input  wire [15:0] img_height,
    input  wire [8:0]  num_in_channels,
    input  wire [4:0]  out_shift,
        // Address bits [15:12] = lane index (0-15), [11:0] = offset
    input  wire [15:0] w_bram_addr,
    input  wire [7:0]  w_bram_data,
    input  wire        w_bram_wen,

    // ========== Bias Loading ==========
    input  wire [3:0]  bias_idx,         // Which lane's bias to write
    input  wire signed [15:0] bias_data, // Bias value
    input  wire        bias_wen,         // Write enable

    // ========== Control ==========
    input  wire        start,
    output reg         busy,
    output reg         done,

    // ========== Input Stream (from DMA) ==========
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,

    // ========== Output Stream (to DMA) ==========
    output reg  [7:0]  m_axis_tdata,
    output reg         m_axis_tvalid,
    output reg         m_axis_tlast,
    input  wire        m_axis_tready
);

    // ========================================================================
    // State Machine
    // ========================================================================
    localparam [3:0] S_IDLE       = 4'd0;
    localparam [3:0] S_INIT       = 4'd1;
    localparam [3:0] S_LOAD_KERN  = 4'd2;
    localparam [3:0] S_STREAM     = 4'd3;
    localparam [3:0] S_FLUSH      = 4'd4;
    localparam [3:0] S_NEXT_CH    = 4'd5;
    localparam [3:0] S_BIAS_RELU  = 4'd6;
    localparam [3:0] S_OUTPUT     = 4'd7;
    localparam [3:0] S_DONE       = 4'd8;

    reg [3:0] state;

    // ========================================================================
    // Bias Storage (16 values, one per lane)
    // ========================================================================
    reg signed [15:0] bias_array [0:N_PARALLEL-1];

    // Initialization
    integer bi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (bi = 0; bi < N_PARALLEL; bi = bi + 1)
                bias_array[bi] <= 16'sd0;
        end else if (bias_wen) begin
            bias_array[bias_idx] <= bias_data;
        end
    end

    // ========================================================================
    // Soft reset for window generators
    // ========================================================================
    reg wg_rst_n;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wg_rst_n <= 1'b0;
        else
            wg_rst_n <= (state != S_IDLE) && (state != S_DONE);
    end

    // ========================================================================
    // Internal Pixel Feed
    // ========================================================================
    reg  [7:0]  feed_pixel;
    reg         feed_valid;
    reg  [15:0] feed_col, feed_row;

    // ========================================================================
    // Window Generator (3×3)
    // ========================================================================
    wire [7:0]  wg3_00, wg3_01, wg3_02;
    wire [7:0]  wg3_10, wg3_11, wg3_12;
    wire [7:0]  wg3_20, wg3_21, wg3_22;
    wire        wg3_valid;
    wire [15:0] wg3_col, wg3_row;

    window_gen_3x3 #(.MAX_WIDTH(MAX_WIDTH)) wg3_inst (
        .clk(clk), .rst_n(wg_rst_n),
        .img_width(img_width), .img_height(img_height),
        .pix_in(feed_pixel), .pix_valid(feed_valid & (kernel_mode == 2'b01)),
        .pix_col(feed_col), .pix_row(feed_row),
        .win_00(wg3_00), .win_01(wg3_01), .win_02(wg3_02),
        .win_10(wg3_10), .win_11(wg3_11), .win_12(wg3_12),
        .win_20(wg3_20), .win_21(wg3_21), .win_22(wg3_22),
        .window_valid(wg3_valid), .win_col(wg3_col), .win_row(wg3_row)
    );

    // ========================================================================
    // Window Generator (4×4)
    // ========================================================================
    wire [7:0]  wg4_00, wg4_01, wg4_02, wg4_03;
    wire [7:0]  wg4_10, wg4_11, wg4_12, wg4_13;
    wire [7:0]  wg4_20, wg4_21, wg4_22, wg4_23;
    wire [7:0]  wg4_30, wg4_31, wg4_32, wg4_33;
    wire        wg4_valid;
    wire [15:0] wg4_col, wg4_row;

    window_gen_4x4 #(.MAX_WIDTH(MAX_WIDTH)) wg4_inst (
        .clk(clk), .rst_n(wg_rst_n),
        .img_width(img_width), .img_height(img_height),
        .pix_in(feed_pixel), .pix_valid(feed_valid & (kernel_mode == 2'b10)),
        .pix_col(feed_col), .pix_row(feed_row),
        .win_00(wg4_00), .win_01(wg4_01), .win_02(wg4_02), .win_03(wg4_03),
        .win_10(wg4_10), .win_11(wg4_11), .win_12(wg4_12), .win_13(wg4_13),
        .win_20(wg4_20), .win_21(wg4_21), .win_22(wg4_22), .win_23(wg4_23),
        .win_30(wg4_30), .win_31(wg4_31), .win_32(wg4_32), .win_33(wg4_33),
        .window_valid(wg4_valid), .win_col(wg4_col), .win_row(wg4_row)
    );

    // ========================================================================
    // Window Mux
    // ========================================================================
    reg  [7:0]  mux_w00, mux_w01, mux_w02, mux_w03;
    reg  [7:0]  mux_w10, mux_w11, mux_w12, mux_w13;
    reg  [7:0]  mux_w20, mux_w21, mux_w22, mux_w23;
    reg  [7:0]  mux_w30, mux_w31, mux_w32, mux_w33;
    reg         mux_valid;
    reg  [15:0] mux_col, mux_row;

    always @(*) begin
        mux_w00=0; mux_w01=0; mux_w02=0; mux_w03=0;
        mux_w10=0; mux_w11=0; mux_w12=0; mux_w13=0;
        mux_w20=0; mux_w21=0; mux_w22=0; mux_w23=0;
        mux_w30=0; mux_w31=0; mux_w32=0; mux_w33=0;
        mux_valid=0; mux_col=0; mux_row=0;
        case (kernel_mode)
            2'b00: begin
                mux_w00=feed_pixel; mux_valid=feed_valid;
                mux_col=feed_col; mux_row=feed_row;
            end
            2'b01: begin
                mux_w00=wg3_00; mux_w01=wg3_01; mux_w02=wg3_02;
                mux_w10=wg3_10; mux_w11=wg3_11; mux_w12=wg3_12;
                mux_w20=wg3_20; mux_w21=wg3_21; mux_w22=wg3_22;
                mux_valid=wg3_valid; mux_col=wg3_col; mux_row=wg3_row;
            end
            2'b10: begin
                mux_w00=wg4_00; mux_w01=wg4_01; mux_w02=wg4_02; mux_w03=wg4_03;
                mux_w10=wg4_10; mux_w11=wg4_11; mux_w12=wg4_12; mux_w13=wg4_13;
                mux_w20=wg4_20; mux_w21=wg4_21; mux_w22=wg4_22; mux_w23=wg4_23;
                mux_w30=wg4_30; mux_w31=wg4_31; mux_w32=wg4_32; mux_w33=wg4_33;
                mux_valid=wg4_valid; mux_col=wg4_col; mux_row=wg4_row;
            end
            default: ;
        endcase
    end

    // Stride filtering
    wire stride_accept;
    assign stride_accept = (stride == 2'd1) ? 1'b1 :
                           ((mux_col[0] == 1'b0) && (mux_row[0] == 1'b0));
    wire engine_window_valid;
    assign engine_window_valid = mux_valid & stride_accept;

    // ========================================================================
    // Output position tracking (delayed to match engine latency)
    // ========================================================================
    reg  [15:0] engine_out_col_d, engine_out_row_d;
    reg  [15:0] out_width, out_height;
    reg  [31:0] total_out_pixels;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_out_col_d <= 0;
            engine_out_row_d <= 0;
        end else begin
            engine_out_col_d <= (stride == 2'd2) ? (mux_col >> 1) : mux_col;
            engine_out_row_d <= (stride == 2'd2) ? (mux_row >> 1) : mux_row;
        end
    end

    wire [11:0] engine_out_addr;
    assign engine_out_addr = engine_out_row_d * out_width + engine_out_col_d;

    // ========================================================================
    // Per-Lane Weight Write Enable Decode
    // ========================================================================
    wire [N_PARALLEL-1:0] lane_w_wr_en;
    genvar gwi;
    generate
        for (gwi = 0; gwi < N_PARALLEL; gwi = gwi + 1) begin : w_decode
            assign lane_w_wr_en[gwi] = w_bram_wen && (w_bram_addr[15:12] == gwi[3:0]);
        end
    endgenerate

    // ========================================================================
    // 16 Parallel Processing Lanes
    // ========================================================================
    // Shared control signals (broadcast to all lanes)
    reg  [11:0] w_rd_addr;          // Weight BRAM read address (shared)
    reg  [3:0]  kern_load_idx;      // Kernel register index
    reg         kern_load_en;       // Load enable for engines

    reg         acc_rmw_trigger;    // Accumulator RMW trigger
    reg  [11:0] acc_rmw_addr_r;     // Accumulator write-back address
    reg  [11:0] acc_rd_addr_r;      // Accumulator read address
    reg         first_channel;      // First input channel flag

    reg  [11:0] out_wr_addr_r;      // Bias+ReLU write address (shared)
    reg         out_wr_en_r;        // Bias+ReLU write enable (shared)

    reg  [11:0] out_rd_addr_r;      // Output BRAM read address

    // Per-lane output data (extracted for muxing during S_OUTPUT)
    wire [7:0] lane_out_data [0:N_PARALLEL-1];

    genvar gi;
    generate
        for (gi = 0; gi < N_PARALLEL; gi = gi + 1) begin : lane
            conv_channel_proc #(
                .WBRAM_DEPTH(WBRAM_DEPTH),
                .ABRAM_DEPTH(ABRAM_DEPTH)
            ) proc_inst (
                .clk          (clk),
                .rst_n        (rst_n),

                // Weight write (PS → specific lane)
                .w_wr_addr    (w_bram_addr[11:0]),
                .w_wr_data    (w_bram_data),
                .w_wr_en      (lane_w_wr_en[gi]),

                // Kernel loading (shared address, shared enable)
                .w_rd_addr    (w_rd_addr),
                .kern_load_idx(kern_load_idx),
                .kern_load_en (kern_load_en),

                // Config
                .kernel_mode  (kernel_mode),

                // Window (shared across all lanes)
                .win_00(mux_w00), .win_01(mux_w01), .win_02(mux_w02), .win_03(mux_w03),
                .win_10(mux_w10), .win_11(mux_w11), .win_12(mux_w12), .win_13(mux_w13),
                .win_20(mux_w20), .win_21(mux_w21), .win_22(mux_w22), .win_23(mux_w23),
                .win_30(mux_w30), .win_31(mux_w31), .win_32(mux_w32), .win_33(mux_w33),
                .window_valid (engine_window_valid),

                // Accumulator (shared control, independent data)
                .acc_rd_addr  (acc_rd_addr_r),
                .acc_rmw_trigger(acc_rmw_trigger),
                .acc_rmw_addr (acc_rmw_addr_r),
                .acc_first_ch (first_channel),

                // Bias + ReLU
                .bias_val     (bias_array[gi]),
                .out_shift    (out_shift),
                // Bias + ReLU output
                .out_wr_addr  (out_wr_addr_r),
                .out_wr_en    (out_wr_en_r),

                // Output
                .out_rd_addr  (out_rd_addr_r),
                .out_rd_data  (lane_out_data[gi])
            );
        end
    endgenerate

    // ========================================================================
    // Output Lane Mux — select which lane's data to stream
    // ========================================================================
    reg  [3:0]  out_lane_idx;
    wire [7:0]  cur_lane_out;

    // Dynamic mux (16:1) — synthesizes to LUT mux
    assign cur_lane_out = lane_out_data[out_lane_idx];

    // ========================================================================
    // Counters & FSM Registers
    // ========================================================================
    reg  [8:0]  cur_channel;
    reg  [15:0] pixel_count;
    reg  [15:0] stream_col, stream_row;
    reg  [31:0] total_pixels;
    reg  [4:0]  kern_load_cnt;
    reg  [4:0]  kern_load_slots;
    reg  [15:0] flush_cnt;
    reg  [11:0] bias_relu_cnt;
    reg  [11:0] output_pos_cnt;
    reg         output_rd_pending;

    // Bias+ReLU Pipeline Tracking
    reg  [11:0] br_addr_p1, br_addr_p2;
    reg         br_en_p1, br_en_p2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            br_addr_p1 <= 0;
            br_addr_p2 <= 0;
            br_en_p1   <= 0;
            br_en_p2   <= 0;
            out_wr_addr_r <= 0;
            out_wr_en_r   <= 0;
        end else begin
            // Stage 1 (Issue read from FSM)
            if (state == S_BIAS_RELU && bias_relu_cnt < total_out_pixels) begin
                br_addr_p1 <= bias_relu_cnt;
                br_en_p1   <= 1'b1;
            end else begin
                br_en_p1   <= 1'b0;
            end
            
            // Stage 2 (BRAM reads)
            br_addr_p2 <= br_addr_p1;
            br_en_p2   <= br_en_p1;

            // Stage 3 (Combinational ReLU + BRAM Write)
            out_wr_addr_r <= br_addr_p2;
            out_wr_en_r   <= br_en_p2;
        end
    end

    assign s_axis_tready = (state == S_STREAM);

    // ========================================================================
    // Main FSM
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            cur_channel    <= 0;
            pixel_count    <= 0;
            stream_col     <= 0;
            stream_row     <= 0;
            total_pixels   <= 0;
            out_width      <= 0;
            out_height     <= 0;
            total_out_pixels <= 0;
            kern_load_cnt  <= 0;
            kern_load_idx  <= 0;
            kern_load_en   <= 0;
            kern_load_slots <= 0;
            flush_cnt      <= 0;
            first_channel  <= 1;
            feed_pixel     <= 0;
            feed_valid     <= 0;
            feed_col       <= 0;
            feed_row       <= 0;
            w_rd_addr      <= 0;
            acc_rmw_trigger <= 0;
            acc_rmw_addr_r <= 0;
            acc_rd_addr_r  <= 0;
            bias_relu_cnt  <= 0;
            out_rd_addr_r  <= 0;
            out_lane_idx   <= 0;
            output_pos_cnt <= 0;
            output_rd_pending <= 0;
            m_axis_tdata   <= 0;
            m_axis_tvalid  <= 0;
            m_axis_tlast   <= 0;
        end else begin
            // Default pulses
            done            <= 1'b0;
            kern_load_en    <= 1'b0;
            feed_valid      <= 1'b0;
            acc_rmw_trigger <= 1'b0;
            m_axis_tvalid   <= 1'b0;
            m_axis_tlast    <= 1'b0;

            case (state)

                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        state         <= S_INIT;
                        busy          <= 1'b1;
                        cur_channel   <= 0;
                        first_channel <= 1'b1;
                    end
                end

                S_INIT: begin
                    total_pixels <= img_width * img_height;
                    case (kernel_mode)
                        2'b00: begin
                            // Each input-channel kernel lives in a 16-slot 4x4 BRAM block.
                            kern_load_slots <= 5'd16;
                            out_width  <= (stride==2'd2) ? (img_width>>1) : img_width;
                            out_height <= (stride==2'd2) ? (img_height>>1) : img_height;
                        end
                        2'b01: begin
                            kern_load_slots <= 5'd16;
                            out_width  <= (stride==2'd2) ? ((img_width-2)>>1) : (img_width-2);
                            out_height <= (stride==2'd2) ? ((img_height-2)>>1) : (img_height-2);
                        end
                        2'b10: begin
                            kern_load_slots <= 5'd16;
                            out_width  <= (stride==2'd2) ? ((img_width-3)>>1) : (img_width-3);
                            out_height <= (stride==2'd2) ? ((img_height-3)>>1) : (img_height-3);
                        end
                        default: begin
                            kern_load_slots <= 5'd16;
                            out_width  <= img_width;
                            out_height <= img_height;
                        end
                    endcase
                    state         <= S_LOAD_KERN;
                    kern_load_cnt <= 0;
                end

                S_LOAD_KERN: begin
                    total_out_pixels <= out_width * out_height;

                    // All lanes read from the same 16-slot block for the current input channel.
                    if (kern_load_cnt < 5'd16)
                        w_rd_addr <= cur_channel * 16 + kern_load_cnt[3:0];
                    else
                        w_rd_addr <= cur_channel * 16 + 4'd15;

                    if (kern_load_cnt > 0) begin
                        kern_load_en  <= 1'b1;      // Broadcast to ALL 16 engines
                        kern_load_idx <= kern_load_cnt - 1;
                    end

                    if (kern_load_cnt == kern_load_slots) begin
                        state       <= S_STREAM;
                        pixel_count <= 0;
                        stream_col  <= 0;
                        stream_row  <= 0;
                    end else begin
                        kern_load_cnt <= kern_load_cnt + 1;
                    end
                end

                S_STREAM: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        feed_pixel  <= s_axis_tdata;
                        feed_valid  <= 1'b1;
                        feed_col    <= stream_col;
                        feed_row    <= stream_row;
                        pixel_count <= pixel_count + 1;

                        if (stream_col == img_width - 1) begin
                            stream_col <= 0;
                            stream_row <= stream_row + 1;
                        end else begin
                            stream_col <= stream_col + 1;
                        end

                        if (pixel_count + 1 >= total_pixels)
                            state <= S_FLUSH;
                    end

                    // Accumulator RMW (broadcast to all lanes)
                    if (engine_window_valid) begin
                        acc_rd_addr_r   <= engine_out_addr;
                        acc_rmw_trigger <= 1'b1;
                        acc_rmw_addr_r  <= engine_out_addr;
                    end
                end

                S_FLUSH: begin
                    flush_cnt <= flush_cnt + 1;

                    if (engine_window_valid) begin
                        acc_rd_addr_r   <= engine_out_addr;
                        acc_rmw_trigger <= 1'b1;
                        acc_rmw_addr_r  <= engine_out_addr;
                    end

                    if (flush_cnt >= img_width + 16) begin
                        flush_cnt <= 0;
                        state     <= S_NEXT_CH;
                    end
                end

                S_NEXT_CH: begin
                    first_channel <= 1'b0;
                    if (cur_channel + 1 >= num_in_channels) begin
                        state         <= S_BIAS_RELU;
                        bias_relu_cnt <= 0;
                    end else begin
                        cur_channel   <= cur_channel + 1;
                        state         <= S_LOAD_KERN;
                        kern_load_cnt <= 0;
                    end
                end

                S_BIAS_RELU: begin
                    // Read accumulator for all 16 lanes
                    if (bias_relu_cnt < total_out_pixels) begin
                        acc_rd_addr_r <= bias_relu_cnt;
                        bias_relu_cnt <= bias_relu_cnt + 1;
                    end
                    
                    // Wait for the pipeline to empty (p1, p2, and out_wr_en_r stages)
                    if (bias_relu_cnt >= total_out_pixels && 
                        !br_en_p1 && !br_en_p2 && !out_wr_en_r) begin
                        state           <= S_OUTPUT;
                        out_lane_idx    <= 0;
                        output_pos_cnt  <= 0;
                        output_rd_pending <= 0;
                    end
                end

                S_OUTPUT: begin
                    // Stream all 16 lanes × H_out × W_out pixels via AXI-Stream
                    // Order: lane 0 all pixels, lane 1 all pixels, ..., lane 15 all pixels

                    if (!output_rd_pending && (out_lane_idx < N_PARALLEL)) begin
                        out_rd_addr_r     <= output_pos_cnt;
                        output_rd_pending <= 1'b1;
                    end else if (output_rd_pending && m_axis_tready) begin
                        m_axis_tdata  <= cur_lane_out;
                        m_axis_tvalid <= 1'b1;
                        output_rd_pending <= 1'b0;

                        // Advance position
                        if (output_pos_cnt + 1 >= total_out_pixels) begin
                            output_pos_cnt <= 0;
                            // Check if this was the last lane
                            if (out_lane_idx + 1 >= N_PARALLEL) begin
                                m_axis_tlast <= 1'b1;
                                state <= S_DONE;
                            end else begin
                                out_lane_idx <= out_lane_idx + 1;
                            end
                        end else begin
                            output_pos_cnt <= output_pos_cnt + 1;
                        end
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
