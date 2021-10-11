import torch
import torch.nn as nn


class ASPP_feat1(nn.Module):
    def __init__(self, channels, features):
        super(ASPP_feat1, self).__init__()
        conv1 = []
        conv1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=1, padding=0))
        # conv1.append(nn.BatchNorm2d(features))
        # conv1.append(nn.PReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv6 = []
        conv6.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=6, dilation=6,bias=False))
        # conv6.append(nn.BatchNorm2d(features))
        # conv6.append(nn.PReLU())
        self.conv6 = nn.Sequential(*conv6)

        conv12 = []
        conv12.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=12, dilation=12, bias=False))
        # conv12.append(nn.BatchNorm2d(features))
        # conv12.append(nn.PReLU())
        self.conv12 = nn.Sequential(*conv12)

        conv18 = []
        conv18.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=18, dilation=18, bias=False))
        # conv18.append(nn.BatchNorm2d(features))
        # conv18.append(nn.PReLU())
        self.conv18 = nn.Sequential(*conv18)

        platlayer = []
        # platlayer.append(nn.BatchNorm2d(features))
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, bias=False))
        self.platlayer = nn.Sequential(*platlayer)

        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.conv1(x)
        d6 = self.conv6(x)
        d12 = self.conv12(x)
        d18 = self.conv18(x)
        dcat = torch.cat((d1, d6, d12, d18), 1)
        dilatedcat = self.feat2feat(dcat)
        out = self.feat2ch(self.platlayer(dilatedcat)) + x
        return out

class ASPP_feat(nn.Module):
    def __init__(self, channels, features):
        super(ASPP_feat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2, bias=False)

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4, bias=False)

        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=8, dilation=8, bias=False)

        platlayer = []
        # platlayer.append(nn.BatchNorm2d(features))
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        self.platlayer = nn.Sequential(*platlayer)

        self.flatten = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = torch.cat((d1, d2, d4, d8), 1)
        dilatedcat = self.flatten(dcat)
        out = self.feat2ch(self.platlayer(dilatedcat)) + x

        return out

class RCA_Block(nn.Module):
    def __init__(self, features):
        super(RCA_Block, self).__init__()
        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        self.firstblock = nn.Sequential(*firstblock)

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

    def forward(self, x):
        residual = x
        data = self.firstblock(x)
        ch_data = self.cab(data) * data
        out = ch_data + residual

        return out

class SCA_Block(nn.Module):
    def __init__(self, features):
        super(SCA_Block, self).__init__()

        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        self.firstblock = nn.Sequential(*firstblock)

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

        platlayer = []
        # platlayer.append(nn.BatchNorm2d(features))
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, bias=False))
        self.platlayer = nn.Sequential(*platlayer)

        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=6, dilation=6,bias=False)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=12, dilation=12, bias=False)
        self.conv18 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=18, dilation=18,bias=False)
        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        d1 = self.conv1(x)
        d6 = self.conv6(x)
        d12 = self.conv12(x)
        d18 = self.conv18(x)
        dcat = torch.cat((d1, d6, d12, d18), 1)
        dilatedcat = self.feat2feat(dcat)
        asppdata = self.platlayer(dilatedcat)

        data = self.firstblock(x)
        ch_data = self.cab(data) * data * asppdata
        out = ch_data + residual

        return out


class ChannelAttention(nn.Module):
    def __init__(self, features):
        super(ChannelAttention, self).__init__()
        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

    def forward(self, x):
        out = self.cab(x) * x

        return out

class RG(nn.Module):
    def __init__(self, features):
        super(RG, self).__init__()
        features = 64
        kernel_size = 3
        padding = 1

        block1 = []
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        # block1.append(nn.BatchNorm2d(features*2))
        block1.append(nn.PReLU())
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        self.block1 = nn.Sequential(*block1)
        self.conv32 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        head1 = self.block1(x) + x
        body1 = self.block1(head1) + head1
        tail1 = self.block1(body1) + body1

        head2 = self.block1(tail1) + tail1
        body2 = self.block1(head2) + head2
        tail2 = self.block1(body2) + body2

        head3 = self.block1(tail2) + tail2
        body3 = self.block1(head3) + head3
        tail3 = self.block1(body3) + body3
        out = self.conv32(tail3) + residual

        return out

class make_dense(nn.Module):
      def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.prelu = nn.PReLU()
      def forward(self, x):
        out = self.prelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
      def __init__(self, features, nDenselayer):
        super(RDB, self).__init__()
        nChannels_ = features
        growthRate = features // 2
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, features, kernel_size=1, padding=0, bias=False)
      def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class ResidudalBlock(nn.Module):
    def __init__(self, features):
        super(ResidudalBlock, self).__init__()

        block1 = []
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        block1.append(nn.PReLU())
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))

        self.block1 = nn.Sequential(*block1)

    def forward(self, x):
        residual = x

        out = self.block1(x) + residual

        return out

##########################################################################
class edgemodule(nn.Module):
    def __init__(self, channels, features):
        super(edgemodule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=3, dilation=3,
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4,
                               bias=False)

        self.prelu = nn.PReLU()
        self.flatten = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0, bias=False)

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        edgeblock.append(nn.PReLU())
        for _ in range(4):
            edgeblock.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            edgeblock.append(nn.PReLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False))

        self.edgeblock = nn.Sequential(*edgeblock)


    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d3 = self.conv3(x)
        d4 = self.conv4(x)

        convcat = self.prelu(torch.cat([d1, d2, d3, d4], 1))
        flatdata = self.flatten(convcat)

        edge = self.edgeblock(flatdata)

        return edge

class feat_extractor(nn.Module):
    def __init__(self, channels, features):
        super(feat_extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2)

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4)

        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=8, dilation=8)

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.ch4feat = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x, edge):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = self.feat2feat(torch.cat([d1, d2, d4, d8], 1))

        dout = self.feat2ch(self.platlayer(dcat))

        out = self.ch4feat(torch.cat([x, edge, x+edge, dout], 1))

        return out

