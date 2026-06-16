// ============================================================================
// CNN Pipeline Top — AXI-wrapped 16-parallel CNN accelerator
//
// Pure Verilog — Compatible with Vivado 2025.1
// Target: Kria KV260 (xck26-sfvc784-2LV-c)
//
// Register Map (AXI-Lite, offset = reg# × 4):
//   Reg 0  [0x00]  CTRL           — bit[0]=start (write 1 to pulse)
//   Reg 1  [0x04]  KERNEL_MODE    — bits[1:0]: 0=1×1, 1=3×3, 2=4×4
//   Reg 2  [0x08]  STRIDE         — bits[1:0]: 1 or 2
//   Reg 3  [0x0C]  IMG_WIDTH      — bits[15:0]
//   Reg 4  [0x10]  IMG_HEIGHT     — bits[15:0]
//   Reg 5  [0x14]  NUM_IN_CH      — bits[8:0]: input channels (1-256)
//   Reg 6  [0x18]  W_BRAM_ADDR    — bits[15:0]: [15:12]=lane, [11:0]=offset
//   Reg 7  [0x1C]  W_BRAM_DATA    — bits[7:0]: weight data
//   Reg 8  [0x20]  W_BRAM_WEN     — bit[0]: write enable (auto-clears)
//   Reg 9  [0x24]  BIAS_IDX       — bits[3:0]: lane index for bias
//   Reg 10 [0x28]  BIAS_DATA      — bits[15:0]: signed bias value
//   Reg 11 [0x2C]  BIAS_WEN       — bit[0]: bias write enable (auto-clears)
//   Reg 12 [0x30]  STATUS (read)  — bit[0]=busy, bit[1]=done_latched
//   Reg 13 [0x34]  OUT_SHIFT      — bits[4:0]: arithmetic right shift before clamp
//   Reg 14 [0x38]  RESERVED
//   Reg 15 [0x3C]  RESERVED
// ============================================================================

`timescale 1ns / 1ps

module cnn_pipeline_top #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 6
) (
    input  wire        aclk,
    input  wire        aresetn,

    // ========== AXI-Lite Slave ==========
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire [2:0]                         s_axi_awprot,
    input  wire                               s_axi_awvalid,
    output reg                                s_axi_awready,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]      s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]  s_axi_wstrb,
    input  wire                               s_axi_wvalid,
    output reg                                s_axi_wready,
    output reg  [1:0]                         s_axi_bresp,
    output reg                                s_axi_bvalid,
    input  wire                               s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]      s_axi_araddr,
    input  wire [2:0]                         s_axi_arprot,
    input  wire                               s_axi_arvalid,
    output reg                                s_axi_arready,
    output reg  [C_S_AXI_DATA_WIDTH-1:0]      s_axi_rdata,
    output reg  [1:0]                         s_axi_rresp,
    output reg                                s_axi_rvalid,
    input  wire                               s_axi_rready,

    // ========== AXI-Stream Slave (from DMA MM2S) ==========
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // ========== AXI-Stream Master (to DMA S2MM) ==========
    output wire [7:0]  m_axis_tdata,
    output wire        m_axis_tvalid,
    output wire        m_axis_tlast,
    input  wire        m_axis_tready
);

    // ========================================================================
    // AXI-Lite Register File
    // ========================================================================
    reg [31:0] slv_reg [0:15];

    reg aw_done, w_done;
    reg [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_latched;

    // Write channel
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            aw_done <= 1'b0;
            w_done  <= 1'b0;
            aw_addr_latched <= 0;
        end else begin
            if (!aw_done && s_axi_awvalid) begin
                s_axi_awready   <= 1'b1;
                aw_addr_latched <= s_axi_awaddr;
                aw_done         <= 1'b1;
            end else
                s_axi_awready <= 1'b0;

            if (!w_done && s_axi_wvalid) begin
                s_axi_wready <= 1'b1;
                w_done       <= 1'b1;
            end else
                s_axi_wready <= 1'b0;

            if (aw_done && w_done) begin
                begin : wr_blk
                    reg [3:0] idx;
                    idx = aw_addr_latched[5:2];
                    if (idx != 4'd12)  // STATUS is read-only
                        slv_reg[idx] <= s_axi_wdata;
                end
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
                aw_done      <= 1'b0;
                w_done       <= 1'b0;
            end

            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    // Read channel
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 0;
            s_axi_rresp   <= 2'b00;
        end else begin
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                begin : rd_blk
                    reg [3:0] idx;
                    idx = s_axi_araddr[5:2];
                    if (idx == 4'd12)
                        s_axi_rdata <= {30'd0, done_latched, compute_busy};
                    else
                        s_axi_rdata <= slv_reg[idx];
                end
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
            end else begin
                s_axi_arready <= 1'b0;
                if (s_axi_rvalid && s_axi_rready)
                    s_axi_rvalid <= 1'b0;
            end
        end
    end

    // ========================================================================
    // Configuration extraction
    // ========================================================================
    wire [1:0]  cfg_kernel_mode = slv_reg[1][1:0];
    wire [1:0]  cfg_stride      = slv_reg[2][1:0];
    wire [15:0] cfg_img_width   = slv_reg[3][15:0];
    wire [15:0] cfg_img_height  = slv_reg[4][15:0];
    wire [8:0]  cfg_num_in_ch   = slv_reg[5][8:0];
    wire [15:0] cfg_w_addr      = slv_reg[6][15:0];
    wire [7:0]  cfg_w_data      = slv_reg[7][7:0];
    wire        cfg_w_wen       = slv_reg[8][0];
    wire [3:0]  cfg_bias_idx    = slv_reg[9][3:0];
    wire signed [15:0] cfg_bias_data = slv_reg[10][15:0];
    wire        cfg_bias_wen    = slv_reg[11][0];
    wire [4:0]  cfg_out_shift   = slv_reg[13][4:0];
        // Start pulse
    reg start_r, start_r2;
    wire start_pulse;
    always @(posedge aclk) begin
        if (!aresetn) begin
            start_r  <= 1'b0;
            start_r2 <= 1'b0;
        end else begin
            start_r  <= slv_reg[0][0];
            start_r2 <= start_r;
        end
    end
    assign start_pulse = start_r & ~start_r2;

    // Auto-clear write enables
    reg w_wen_prev, b_wen_prev;
    always @(posedge aclk) begin
        if (!aresetn) begin
            w_wen_prev <= 0;
            b_wen_prev <= 0;
        end else begin
            w_wen_prev <= cfg_w_wen;
            b_wen_prev <= cfg_bias_wen;
            if (w_wen_prev) slv_reg[8]  <= 32'h0;
            if (b_wen_prev) slv_reg[11] <= 32'h0;
        end
    end

    wire w_wen_pulse = cfg_w_wen & ~w_wen_prev;
    wire b_wen_pulse = cfg_bias_wen & ~b_wen_prev;

    // ========================================================================
    // Status
    // ========================================================================
    wire compute_busy, compute_done;
    reg  done_latched;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            done_latched <= 1'b0;
        else if (compute_done)
            done_latched <= 1'b1;
        else if (start_pulse)
            done_latched <= 1'b0;
    end

    // ========================================================================
    // Output — direct passthrough from compute unit
    // (MaxPool is done in software on the ARM PS)
    // ========================================================================
    wire [7:0] compute_tdata;
    wire       compute_tvalid;
    wire       compute_tlast;

    assign m_axis_tdata  = compute_tdata;
    assign m_axis_tvalid = compute_tvalid;
    assign m_axis_tlast  = compute_tlast;

    // ========================================================================
    // Compute Unit (16-parallel)
    // ========================================================================
    cnn_compute_unit #(
        .N_PARALLEL  (16),
        .MAX_WIDTH   (128),
        .WBRAM_DEPTH (4096),
        .ABRAM_DEPTH (4096)
    ) compute_inst (
        .clk             (aclk),
        .rst_n           (aresetn),
        .kernel_mode     (cfg_kernel_mode),
        .stride          (cfg_stride),
        .img_width       (cfg_img_width),
        .img_height      (cfg_img_height),
        .num_in_channels (cfg_num_in_ch),
        .out_shift       (cfg_out_shift),
                .w_bram_addr     (cfg_w_addr),
        .w_bram_data     (cfg_w_data),
        .w_bram_wen      (w_wen_pulse),
        .bias_idx        (cfg_bias_idx),
        .bias_data       (cfg_bias_data),
        .bias_wen        (b_wen_pulse),
        .start           (start_pulse),
        .busy            (compute_busy),
        .done            (compute_done),
        .s_axis_tdata    (s_axis_tdata),
        .s_axis_tvalid   (s_axis_tvalid),
        .s_axis_tready   (s_axis_tready),
        .m_axis_tdata    (compute_tdata),
        .m_axis_tvalid   (compute_tvalid),
        .m_axis_tlast    (compute_tlast),
        .m_axis_tready   (m_axis_tready)
    );

endmodule
