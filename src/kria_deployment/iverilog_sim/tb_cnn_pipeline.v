`timescale 1ns / 1ps

module tb_cnn_pipeline;

    // Clock and Reset
    reg aclk;
    reg aresetn;

    // AXI-Lite Slave
    reg  [5:0]  s_axi_awaddr;
    reg  [2:0]  s_axi_awprot;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;
    reg  [5:0]  s_axi_araddr;
    reg  [2:0]  s_axi_arprot;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    // AXI-Stream Slave
    reg  [7:0]  s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tlast;

    // AXI-Stream Master
    wire [7:0]  m_axis_tdata;
    wire        m_axis_tvalid;
    wire        m_axis_tlast;
    reg         m_axis_tready;

    // Instantiate the DUT
    cnn_pipeline_top dut (
        .aclk          (aclk),
        .aresetn       (aresetn),
        .s_axi_awaddr  (s_axi_awaddr),
        .s_axi_awprot  (s_axi_awprot),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_axi_wdata),
        .s_axi_wstrb   (s_axi_wstrb),
        .s_axi_wvalid  (s_axi_wvalid),
        .s_axi_wready  (s_axi_wready),
        .s_axi_bresp   (s_axi_bresp),
        .s_axi_bvalid  (s_axi_bvalid),
        .s_axi_bready  (s_axi_bready),
        .s_axi_araddr  (s_axi_araddr),
        .s_axi_arprot  (s_axi_arprot),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata   (s_axi_rdata),
        .s_axi_rresp   (s_axi_rresp),
        .s_axi_rvalid  (s_axi_rvalid),
        .s_axi_rready  (s_axi_rready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tlast  (m_axis_tlast),
        .m_axis_tready (m_axis_tready)
    );

    // Clock Generation
    initial aclk = 0;
    always #5 aclk = ~aclk; // 100MHz

    // AXI-Lite Write Task
    task axi_write(input [5:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1;
            s_axi_bready  <= 1;
            
            // Wait for awready
            while (!s_axi_awready) @(posedge aclk);
            s_axi_awvalid <= 0;
            
            // Wait for wready (might be already done if they assert same time)
            if (s_axi_wvalid) begin
                while (!s_axi_wready) @(posedge aclk);
                s_axi_wvalid <= 0;
            end
            
            // Wait for bvalid
            while (!s_axi_bvalid) @(posedge aclk);
            s_axi_bready <= 0;
        end
    endtask

    // File descriptors and arrays
    integer f_out, i, l;
    reg [7:0] in_img [0:499999];
    reg [7:0] weights [0:32767];
    reg [15:0] bias [0:15];

    integer img_width = 64;
    integer img_height = 64;
    integer in_channels = 1;
    integer kernel_mode = 1;
    integer stride = 1;
    integer weight_cnt = 9;
    integer tb_out_shift = 0;
    integer in_pixels = 4096;
    
    // Initialize block
    initial begin
        // $dumpfile("sim.vcd");
        // $dumpvars(0, tb_cnn_pipeline);
        
        if ($value$plusargs("IMG_WIDTH=%d", img_width)) $display("IMG_WIDTH=%d", img_width);
        if ($value$plusargs("IMG_HEIGHT=%d", img_height)) $display("IMG_HEIGHT=%d", img_height);
        if ($value$plusargs("IN_CHANNELS=%d", in_channels)) $display("IN_CHANNELS=%d", in_channels);
        if ($value$plusargs("KERNEL_MODE=%d", kernel_mode)) $display("KERNEL_MODE=%d", kernel_mode);
        if ($value$plusargs("STRIDE=%d", stride)) $display("STRIDE=%d", stride);
        if ($value$plusargs("WEIGHT_CNT=%d", weight_cnt)) $display("WEIGHT_CNT=%d", weight_cnt);
        if ($value$plusargs("OUT_SHIFT=%d", tb_out_shift)) $display("OUT_SHIFT=%d", tb_out_shift);
                if ($value$plusargs("IN_PIXELS=%d", in_pixels)) $display("IN_PIXELS=%d", in_pixels);
                
        // Initialize Inputs
        aresetn       = 0;
        s_axi_awaddr  = 0;
        s_axi_awprot  = 0;
        s_axi_awvalid = 0;
        s_axi_wdata   = 0;
        s_axi_wstrb   = 0;
        s_axi_wvalid  = 0;
        s_axi_bready  = 0;
        s_axi_araddr  = 0;
        s_axi_arprot  = 0;
        s_axi_arvalid = 0;
        s_axi_rready  = 0;

        s_axis_tdata  = 0;
        s_axis_tvalid = 0;
        s_axis_tlast  = 0;

        m_axis_tready = 1; // Always ready to receive

        // Hold reset for 100ns
        #100 aresetn = 1;
        #100;
        
        $display("Reading files...");
        // Load files
        $readmemh("image_in.hex", in_img);
        $readmemh("weights.hex", weights);
        $readmemh("biases.hex", bias);

        $display("Configuring registers...");
        // Clear CTRL to 0 to initialize X state
        axi_write(6'h00, 0);
        
        // Reg 3 [0x0C] IMG_WIDTH
        axi_write(6'h0C, img_width);
        // Reg 4 [0x10] IMG_HEIGHT
        axi_write(6'h10, img_height);
        // Reg 1 [0x04] KERNEL_MODE
        axi_write(6'h04, kernel_mode);
        // Reg 2 [0x08] STRIDE
        axi_write(6'h08, stride);
        // Reg 5 [0x14] NUM_IN_CH
        axi_write(6'h14, in_channels);
        // Reg 13 [0x34] OUT_SHIFT
        axi_write(6'h34, tb_out_shift);
        
        $display("Loading biases and weights...");
        // Initialize biases
        for (i=0; i<16; i=i+1) begin
            axi_write(6'h24, i);
            axi_write(6'h28, bias[i]);
            axi_write(6'h2C, 1);
        end

        // Load Weights
        for (i = 0; i < weight_cnt; i = i + 1) begin
            for (l = 0; l < 16; l = l + 1) begin
                axi_write(6'h18, {l[3:0], i[11:0]});   // Address
                axi_write(6'h1C, weights[l * weight_cnt + i]);        // Data
                axi_write(6'h20, 1);                 // Write Enable
            end
        end

        $display("Sending start pulse...");
        // Start processing
        axi_write(6'h00, 1);
        repeat(5) @(posedge aclk);

        $display("Internal Reg Start: %b, \nState: %d, \nKernel mode: %d, \nImage WxH: %dx%d(real: %dx%d), \nLoad Cnt: %d", dut.slv_reg[0][0], dut.compute_inst.state, dut.compute_inst.kernel_mode, dut.compute_inst.img_width, dut.compute_inst.img_height, dut.cfg_img_width, dut.cfg_img_height, dut.compute_inst.kern_load_cnt);

        $display("Streaming image data...");
        // Start Streaming Input Image Data
        for (i = 0; i < in_pixels; i = i + 1) begin
            s_axis_tvalid <= 1;
            s_axis_tdata  <= in_img[i];
            s_axis_tlast  <= (i == in_pixels - 1);
            
            @(posedge aclk);
            while (!s_axis_tready) @(posedge aclk);
            
            if (i % 512 == 0) $display("Sent %d pixels", i);
        end
        
        s_axis_tvalid <= 0;
        s_axis_tlast  <= 0;
        
        $display("Finished streaming. Waiting for complete...");
        $display("Final pixel count in DUT: %d, total needed: %d", dut.compute_inst.pixel_count, dut.compute_inst.total_pixels);

        // Give it time to finish
        #5000000;
        $display("Timeout reached. Exiting gracefully.");
        $fclose(f_out);
        $finish;
    end
    
    // Output capture process
    integer out_count = 0;
    initial begin
        f_out = $fopen("image_out.hex", "w");
        if (f_out == 0) begin
            $display("Failed to open image_out.hex");
            $finish;
        end
    end

    always @(posedge aclk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            $fdisplay(f_out, "%02X", m_axis_tdata);
            out_count = out_count + 1;
            if (m_axis_tlast) begin
                $display("Received tlast. Total elements: %d", out_count);
                // Give it a bit more time to settle
                #100 
                $fclose(f_out);
                $finish;
            end
        end
    end

    // Monitor State Changes
    always @(dut.compute_inst.state) begin
        $display("Time %0t: FSM State changed to %d", $time, dut.compute_inst.state);
    end

endmodule
