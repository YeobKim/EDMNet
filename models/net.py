import torch
import torch.nn as nn
import models.common as common

import torchvision.ops.deform_conv as dc


class _down(nn.Module):
    def __init__(self, nchannel):
        super(_down, self).__init__()
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=2*nchannel, kernel_size=1, stride=1, padding=0)
        self.conv_down = nn.Conv2d(in_channels=nchannel, out_channels=2 * nchannel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.prelu(self.conv_down(x))

        return out

class _up(nn.Module):
    def __init__(self, nchannel):
        super(_up, self).__init__()
        self.prelu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nchannel, out_channels=nchannel//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.prelu(self.subpixel(x))

        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class RCAB(nn.Module):
    def __init__(self, nchannel):
        super(RCAB, self).__init__()
        self.RCAB = common.RCA_Block(nchannel)
    def forward(self, x):
        out = self.RCAB(x)

        return out

class deform_conv(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(deform_conv, self).__init__()
        groups = 8
        kernel_size = 3

        self.prelu = nn.PReLU()
        self.offset_conv1 = nn.Conv2d(features, 2*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv1 = dc.DeformConv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=groups)

    def forward(self, x):
        # deform conv
        offset1 = self.prelu(self.offset_conv1(x))
        out = self.deconv1(x, offset1)

        return out

class DAB(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(DAB, self).__init__()
        groups = 8
        kernel_size = 3

        self.prelu = nn.PReLU()
        self.offset_conv1 = nn.Conv2d(features, 2*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv1 = dc.DeformConv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=groups)
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        residual = x
        # deform conv
        offset1 = self.prelu(self.offset_conv1(x))
        feat_deconv1 = self.deconv1(x, offset1)

        # attention
        atten_conv = self.conv(x)
        atten_feat = self.softmax(atten_conv)

        out = atten_feat * feat_deconv1
        out = out + residual

        return out

class DCCAB(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(DCCAB, self).__init__()
        self.dab = DAB(features)
        self.cab = RCAB(features)

    def forward(self, x):
        residual = x
        # deform conv
        dabdata = self.dab(x)
        cabdata = self.cab(dabdata)

        out = cabdata + residual

        return out


class oam(nn.Module): # Object Attention Module
    def __init__(self, nchannel):
        super(oam, self).__init__()
        features = nchannel

        self.prelu = nn.PReLU()
        self.encode1_conv = RCAB(features)
        # self.encode1_conv = self.make_layer(RG, features)
        self.down1 = self.make_layer(_down, features)
        self.deconv1 = DCCAB(features)

        self.encode2_conv = RCAB(features*2)
        self.down2 = self.make_layer(_down, features*2)
        self.deconv2 = DCCAB(features*2)

        self.encode3_conv = RCAB(features*4)
        self.deconv3 = DCCAB(features*4)

        self.up1 = self.make_layer(_up, features*8)
        self.decode1_conv = RCAB(features*4)

        self.up2 = self.make_layer(_up, features*4)
        self.decode2_conv = RCAB(features*2)

        self.convf = nn.Conv2d(in_channels=nchannel * 2, out_channels=nchannel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=3, padding=1)
        self.flat_flatten = nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # U-Net
        out1 = self.encode1_conv(x)
        bridge1 = self.deconv1(out1)
        out1_down = self.down1(out1) #64

        out2 = self.encode2_conv(out1_down)
        bridge2 = self.deconv2(out2)
        out2_down = self.down2(out2) #128

        out3 = self.encode3_conv(out2_down)
        bridge3 = self.deconv3(out3)

        catout1 = torch.cat([out3, bridge3], 1) #256

        upout1 = self.up1(catout1) #128
        catout2 = torch.cat([bridge2, upout1], 1)

        out_11 = self.decode1_conv(catout2)
        upout2 = self.up2(out_11)
        catout4 = torch.cat([bridge1, upout2], 1)

        out_21 = self.decode2_conv(catout4)
        out = self.convf(out_21)

        out = out + residual

        return out

class RDFB(nn.Module): # Residual Feature Aggregation
    def __init__(self, nchannel):
        super(RDFB, self).__init__()
        features = nchannel

        self.prelu = nn.PReLU()

        self.RB = common.ResidudalBlock(features)

        self.convf = nn.Conv2d(in_channels=nchannel * 4, out_channels=nchannel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=3, padding=1)
        self.flat_flatten = nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # local residual block
        res1 = x + self.RB(x)
        res2 = x + self.RB(res1)
        res3 = x + self.RB(res2)
        res4 = self.RB(res3)
        res_cat = torch.cat([res1, res2, res3, res4], 1)

        out = self.convf(res_cat) + residual

        return out

class edmnet(nn.Module):
    def __init__(self, channels=3, features=128):
        super(edmnet, self).__init__()

        self.ASPP = common.ASPP_feat(channels, features)
        self.feature_extract = common.RCA_Block(features)

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=1, padding=0, bias=False))
        edgeblock.append(nn.PReLU())
        for _ in range(4):
            edgeblock.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            edgeblock.append(nn.PReLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0, bias=False))
        self.edgeblock = nn.Sequential(*edgeblock)

        # self.edgeblock = edge_module(channels, features)

        self.unet = oam(features)
        self.pipeline = RDFB(features//2)

        self.conv2 = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)
        self.conv2ch = nn.Conv2d(in_channels=features//2, out_channels=channels, kernel_size=1,padding=0)
        self.convf = nn.Conv2d(in_channels=features//4, out_channels=features//2, kernel_size=1,padding=0)
        self.feat2feat = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1,padding=0)
        self.conv_down = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=2, padding=1,bias=False)
        self.conv_up = nn.PixelShuffle(2)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # Data Generator
        edge = x - self.edgeblock(x)
        aspp_feat = self.ASPP(x)
        input_data = self.conv2(torch.cat([x, edge, x+edge, aspp_feat], 1))

        down_input = self.conv_down(input_data)

        H = down_input.size(2)
        W = down_input.size(3)

        # Two Patches for stage1
        x2_img1 = down_input[:, :, 0:int(H / 2), :]
        x2_img2 = down_input[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1_top_img1 = x2_img1[:, :, :, 0:int(W / 2)]
        x1_top_img2 = x2_img1[:, :, :, int(W / 2):W]
        x1_bot_img1 = x2_img2[:, :, :, 0:int(W / 2)]
        x1_bot_img2 = x2_img2[:, :, :, int(W / 2):W]

        # U-Net Stage1
        stage1_out1 = self.unet(x1_top_img1)
        stage1_out2 = self.unet(x1_top_img2)
        stage1_out3 = self.unet(x1_bot_img1)
        stage1_out4 = self.unet(x1_bot_img2)

        # Concat deep features
        recon_stage1_1 = torch.cat([stage1_out1, stage1_out2], 3)
        recon_stage1_2 = torch.cat([stage1_out3, stage1_out4], 3)

        # U-Net Stage2
        stage2_input1 = self.feat2feat(torch.cat([recon_stage1_1, x2_img1], 1))
        stage2_input2 = self.feat2feat(torch.cat([recon_stage1_2, x2_img2], 1))

        stage2_out1 = self.unet(stage2_input1)
        stage2_out2 = self.unet(stage2_input2)

        # Concat deep features
        recon_stage2 = torch.cat([stage2_out1, stage2_out2], 2)

        # U-Net Stage3
        stage3_input = self.feat2feat(torch.cat([down_input, recon_stage2], 1))
        stage3_out = self.unet(stage3_input)

        # BottleNeck
        up_output = self.conv_up(stage3_out)
        output = self.convf(up_output)

        # RDFB Pipeline
        pipeout1 = self.pipeline(output)
        pipeout2 = self.pipeline(pipeout1) + output

        out_ch = self.conv2ch(pipeout2)
        out = out_ch + residual

        return out, edge