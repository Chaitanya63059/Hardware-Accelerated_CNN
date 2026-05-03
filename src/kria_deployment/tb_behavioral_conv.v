`timescale 1ns / 1ps

module tb_behavioral_conv;
    integer img_width = 64;
    integer img_height = 64;
    integer in_channels = 1;
    integer kernel_mode = 1;
    integer stride = 1;
    integer out_shift = 0;
    integer out_width;
    integer out_height;

    reg [7:0] in_img [0:799999];
    reg [7:0] weights [0:65535];
    reg signed [15:0] bias [0:15];

    integer f_out;
    integer lane;
    integer oh;
    integer ow;
    integer ic;
    integer slot;
    integer kh;
    integer kw;
    integer in_row;
    integer in_col;
    integer in_addr;
    integer w_addr;
    integer acc;
    integer shifted;
    integer pix;
    integer w_signed;

    initial begin
        if ($value$plusargs("IMG_WIDTH=%d", img_width)) begin end
        if ($value$plusargs("IMG_HEIGHT=%d", img_height)) begin end
        if ($value$plusargs("IN_CHANNELS=%d", in_channels)) begin end
        if ($value$plusargs("KERNEL_MODE=%d", kernel_mode)) begin end
        if ($value$plusargs("STRIDE=%d", stride)) begin end
        if ($value$plusargs("OUT_SHIFT=%d", out_shift)) begin end

        $readmemh("image_in.hex", in_img);
        $readmemh("weights.hex", weights);
        $readmemh("biases.hex", bias);

        case (kernel_mode)
            0: begin
                out_width = (stride == 2) ? (img_width >> 1) : img_width;
                out_height = (stride == 2) ? (img_height >> 1) : img_height;
            end
            1: begin
                out_width = (stride == 2) ? ((img_width - 2) >> 1) : (img_width - 2);
                out_height = (stride == 2) ? ((img_height - 2) >> 1) : (img_height - 2);
            end
            2: begin
                out_width = (stride == 2) ? ((img_width - 3) >> 1) : (img_width - 3);
                out_height = (stride == 2) ? ((img_height - 3) >> 1) : (img_height - 3);
            end
            default: begin
                out_width = img_width;
                out_height = img_height;
            end
        endcase

        f_out = $fopen("image_out.hex", "w");
        if (f_out == 0) begin
            $display("ERROR: failed to open image_out.hex");
            $finish;
        end

        for (lane = 0; lane < 16; lane = lane + 1) begin
            for (oh = 0; oh < out_height; oh = oh + 1) begin
                for (ow = 0; ow < out_width; ow = ow + 1) begin
                    acc = bias[lane];
                    for (ic = 0; ic < in_channels; ic = ic + 1) begin
                        for (slot = 0; slot < 16; slot = slot + 1) begin
                            kh = slot / 4;
                            kw = slot % 4;
                            in_row = oh * stride + kh;
                            in_col = ow * stride + kw;
                            if (in_row < img_height && in_col < img_width) begin
                                in_addr = ic * img_height * img_width + in_row * img_width + in_col;
                                w_addr = lane * in_channels * 16 + ic * 16 + slot;
                                pix = in_img[in_addr];
                                w_signed = weights[w_addr];
                                if (w_signed >= 128)
                                    w_signed = w_signed - 256;
                                acc = acc + (pix * w_signed);
                            end
                        end
                    end

                    shifted = acc >>> out_shift;
                    if (shifted < 0)
                        $fdisplay(f_out, "00");
                    else if (shifted > 127)
                        $fdisplay(f_out, "7F");
                    else
                        $fdisplay(f_out, "%02X", shifted[7:0]);
                end
            end
        end

        $fclose(f_out);
        $finish;
    end
endmodule
