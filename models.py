import torch
import torch.nn as nn
# import SiLU
import common_prelu

class _down(nn.Module):
    def __init__(self, nchannel):
        super(_down, self).__init__()
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=2*nchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.prelu(self.conv(out))

        return out

class _up(nn.Module):
    def __init__(self, nchannel):
        super(_up, self).__init__()
        self.prelu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.prelu(self.conv(x))
        out = self.subpixel(out)
        return out


class RCAB(nn.Module):
    def __init__(self, nchannel=64):
        super(RCAB, self).__init__()
        self.RCAB = common_prelu.RCA_Block(nchannel)
    def forward(self, x):
        out = self.RCAB(x)

        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class DilatedResBlock(nn.Module):
    def __init__(self, nchannel):
        super(DilatedResBlock, self).__init__()
        features = nchannel
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=2, dilation=2,
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=4, dilation=4,
                               bias=False)
        self.conv8 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=8, dilation=8,
                               bias=False)

        self.prelu = nn.PReLU()
        self.flatten = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0, bias=False)

        resblock = []
        resblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        resblock.append(nn.PReLU())
        resblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        self.RB = nn.Sequential(*resblock)

        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

        self.compress = ChannelPool()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv4 = self.conv4(x)
        conv8 = self.conv8(x)
        convcat = self.prelu(torch.cat([conv1, conv2, conv4, conv8], 1))

        flatdata = self.flatten(convcat)

        sp_data = self.sab(self.compress(flatdata)) * flatdata

        out = self.RB(sp_data) + x

        return out

class DCAR_Unet(nn.Module):
    def __init__(self, nchannel):
        super(DCAR_Unet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        self.prelu = nn.PReLU()
        self.encode1_conv1 = self.make_layer(RCAB, 64)
        self.encode1_conv2 = self.make_layer(RCAB, 64)
        self.down1 = self.make_layer(_down, 64)

        self.encode2_conv1 = self.make_layer(RCAB, 128)
        self.encode2_conv2 = self.make_layer(RCAB, 128)
        self.down2 = self.make_layer(_down, 128)

        self.encode3_conv1 = self.make_layer(RCAB, 256)
        self.encode3_conv2 = self.make_layer(RCAB, 256)

        self.RDB = common_prelu.RDB(256, 8)
        self.DRB = self.make_layer(DilatedResBlock, 256)

        self.up1 = self.make_layer(_up, 512)
        self.decode1_conv1 = self.make_layer(RCAB, 256)
        self.decode1_conv2 = self.make_layer(RCAB, 256)

        self.up2 = self.make_layer(_up, 256)
        self.decode2_conv1 = self.make_layer(RCAB, 128)
        self.decode2_conv2 = self.make_layer(RCAB, 128)

        self.convf = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # U-Net
        out1_1 = self.encode1_conv1(x)
        conc1 = self.encode1_conv2(out1_1)
        out2 = self.down1(conc1) #128

        out2_1 = self.encode2_conv1(out2)
        conc2 = self.encode2_conv2(out2_1)
        out3 = self.down2(conc2) #256

        out3_1 = self.encode3_conv1(out3)
        out3_2 = self.encode3_conv2(out3_1)

        drb1 = self.DRB(out3_2)
        drb2 = self.DRB(drb1)
        drb3 = self.DRB(drb2)
        drb4 = self.DRB(drb3) + out3_2

        catout1 = torch.cat([out3, drb4], 1) #512
        upout1 = self.up1(catout1) #256
        catout2 = torch.cat([conc2, upout1], 1)
        out_11 = self.decode1_conv1(catout2)
        out_12 = self.decode1_conv2(out_11)
        upout2 = self.up2(out_12)
        catout4 = torch.cat([conc1, upout2], 1)
        out_21 = self.decode2_conv1(catout4)
        out_22 = self.decode2_conv2(out_21)
        out = self.prelu(self.convf(out_22))

        out = out + residual

        return out

class MEDNet(nn.Module):
    def __init__(self, channels):
        super(MEDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        self.SCAB = common_prelu.SCAB(features)
        self.RCAB = common_prelu.RCA_Block(features)
        self.featblock = common_prelu.kernelblock(3)
        self.DilatedResBlock = self.make_layer(DilatedResBlock, features)

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=1, padding=0, bias=False))
        edgeblock.append(nn.PReLU())
        for _ in range(4):
            edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            edgeblock.append(nn.PReLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0, bias=False))
        self.edgeblock = nn.Sequential(*edgeblock)

        self.DCAR_Unet = self.make_layer(DCAR_Unet, 64)

        self.convf = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)
        self.conv2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=features * 3, out_channels=features, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # Data Generator
        edge = x - self.edgeblock(x)
        feat = self.featblock(x)
        inputdata = self.conv2(torch.cat([x, edge, x+edge, feat], 1))

        H = inputdata.size(2)
        W = inputdata.size(3)

        # Two Patches for stage1
        x1_img1 = inputdata[:, :, 0:int(H / 2), :]
        x1_img2 = inputdata[:, :, int(H / 2):H, :]

        # U-Net Stage1
        stage1_out1 = self.DCAR_Unet(x1_img1)
        stage1_out2 = self.DCAR_Unet(x1_img2)

        # Concat deep features
        # recon_stage1 = [torch.cat((k, v), 2) for k, v in zip(stage1_out1, stage1_out2)]
        recon_stage1 = torch.cat([stage1_out1, stage1_out2], 2)
        # U-Net Stage2
        stage2_input = inputdata + recon_stage1
        stage2_out = self.DCAR_Unet(stage2_input)

        # BottleNeck
        output = self.conv2ch(stage2_out)

        out = output + residual

        return out, edge

class DCAR_Unet2(nn.Module):
    def __init__(self, nchannel):
        super(DCAR_Unet2, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        self.prelu = nn.PReLU()
        self.encode1_conv1 = self.make_layer(RCAB, 64)
        self.encode1_conv2 = self.make_layer(RCAB, 64)
        self.down1 = self.make_layer(_down, 64)

        self.encode2_conv1 = self.make_layer(RCAB, 128)
        self.encode2_conv2 = self.make_layer(RCAB, 128)
        self.down2 = self.make_layer(_down, 128)

        self.encode3_conv1 = self.make_layer(RCAB, 256)
        self.encode3_conv2 = self.make_layer(RCAB, 256)

        self.RDB = common_prelu.RDB(256, 8)

        self.DRB1 = self.make_layer(DilatedResBlock, 64)
        self.DRB2 = self.make_layer(DilatedResBlock, 128)
        self.DRB3 = self.make_layer(DilatedResBlock, 256)

        self.up1 = self.make_layer(_up, 512)
        self.decode1_conv1 = self.make_layer(RCAB, 256)
        self.decode1_conv2 = self.make_layer(RCAB, 256)

        self.up2 = self.make_layer(_up, 256)
        self.decode2_conv1 = self.make_layer(RCAB, 128)
        self.decode2_conv2 = self.make_layer(RCAB, 128)

        self.convf = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # U-Net
        out1_1 = self.encode1_conv1(x)
        conc1 = self.encode1_conv2(out1_1)
        conc1_drb = self.DRB1(conc1)
        out2 = self.down1(conc1) #128

        out2_1 = self.encode2_conv1(out2)
        conc2 = self.encode2_conv2(out2_1)
        conc2_drb = self.DRB2(conc2)
        out3 = self.down2(conc2) #256

        out3_1 = self.encode3_conv1(out3)
        out3_2 = self.encode3_conv2(out3_1)

        drb = self.DRB3(out3_2)

        catout1 = torch.cat([out3, drb], 1) #512
        upout1 = self.up1(catout1) #256
        catout2 = torch.cat([conc2_drb, upout1], 1)
        out_11 = self.decode1_conv1(catout2)
        out_12 = self.decode1_conv2(out_11)
        upout2 = self.up2(out_12)
        catout4 = torch.cat([conc1_drb, upout2], 1)
        out_21 = self.decode2_conv1(catout4)
        out_22 = self.decode2_conv2(out_21)
        out = self.prelu(self.convf(out_22))

        out = out + residual

        return out

class MEDNet2(nn.Module):
    def __init__(self, channels):
        super(MEDNet2, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        self.SCAB = common_prelu.SCAB(features)
        self.RCAB = common_prelu.RCA_Block(features)
        self.featblock = common_prelu.kernelblock(3)
        self.DilatedResBlock = self.make_layer(DilatedResBlock, features)

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=1, padding=0, bias=False))
        edgeblock.append(nn.PReLU())
        for _ in range(4):
            edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            edgeblock.append(nn.PReLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0, bias=False))
        self.edgeblock = nn.Sequential(*edgeblock)

        self.DCAR_Unet = self.make_layer(DCAR_Unet, 64)

        self.convf = nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)
        self.conv2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=features * 3, out_channels=features, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def make_layer(selfself, block, nchannel):
        layers = []
        layers.append(block(nchannel=nchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        # Data Generator
        edge = x - self.edgeblock(x)
        feat = self.featblock(x)
        inputdata = self.conv2(torch.cat([x, edge, x+edge, feat], 1))

        H = inputdata.size(2)
        W = inputdata.size(3)

        # Two Patches for stage1
        x1_img1 = inputdata[:, :, 0:int(H / 2), :]
        x1_img2 = inputdata[:, :, int(H / 2):H, :]

        # U-Net Stage1
        stage1_out1 = self.DCAR_Unet(x1_img1)
        stage1_out2 = self.DCAR_Unet(x1_img2)

        # Concat deep features
        # recon_stage1 = [torch.cat((k, v), 2) for k, v in zip(stage1_out1, stage1_out2)]
        recon_stage1 = torch.cat([stage1_out1, stage1_out2], 2)
        # U-Net Stage2
        stage2_input = inputdata + recon_stage1
        stage2_out = self.DCAR_Unet(stage2_input)

        # BottleNeck
        output = self.conv2ch(stage2_out)

        out = output + residual

        return out, edge