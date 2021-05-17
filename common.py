import torch
import torch.nn as nn
# import SiLU
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class ASPP(nn.Module):
    def __init__(self, features):
        super(ASPP, self).__init__()
        conv1 = []
        conv1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0))
        # conv1.append(nn.BatchNorm2d(features))
        # conv1.append(nn.PReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv6 = []
        conv6.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=6, dilation=6,bias=False))
        # conv6.append(nn.BatchNorm2d(features))
        # conv6.append(nn.PReLU())
        self.conv6 = nn.Sequential(*conv6)

        conv12 = []
        conv12.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=12, dilation=12, bias=False))
        # conv12.append(nn.BatchNorm2d(features))
        # conv12.append(nn.PReLU())
        self.conv12 = nn.Sequential(*conv12)

        conv18 = []
        conv18.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=18, dilation=18, bias=False))
        # conv18.append(nn.BatchNorm2d(features))
        # conv18.append(nn.PReLU())
        self.conv18 = nn.Sequential(*conv18)

        platlayer = []
        # platlayer.append(nn.BatchNorm2d(features))
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, bias=False))
        self.platlayer = nn.Sequential(*platlayer)

        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.conv1(x)
        d6 = self.conv6(x)
        d12 = self.conv12(x)
        d18 = self.conv18(x)
        dcat = torch.cat((d1, d6, d12, d18), 1)
        dilatedcat = self.feat2feat(dcat)
        out = self.platlayer(dilatedcat)
        return out

class kernelblock(nn.Module):
    def __init__(self, channels):
        super(kernelblock, self).__init__()
        features = 64
        self.kernel1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=1, padding=0)
        self.kernel3 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)
        self.kernel5 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=5, padding=2)
        self.kernel7 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=7, padding=3)

        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x):
        k1 = self.prelu(self.kernel1(x))
        k3 = self.prelu(self.kernel3(x))
        k5 = self.prelu(self.kernel5(x))
        k7 = self.prelu(self.kernel7(x))
        kernelcat = torch.cat((k1, k3, k5, k7), 1)
        kernelcat = self.feat2feat(kernelcat)
        out = self.prelu(kernelcat)
        out = self.conv2(out)
        out = self.conv(self.prelu(out))
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

class ARCA_Block(nn.Module):
    def __init__(self, features):
        super(ARCA_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=6, dilation=6,bias=False)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=12, dilation=12, bias=False)
        self.conv18 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=18, dilation=18, bias=False)

        self.RCAB = RCA_Block(features)
        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0, bias=False))
        self.platlayer = nn.Sequential(*platlayer)

        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x):
        # ASPP Block
        d1 = self.conv1(x)
        d6 = self.conv6(x)
        d12 = self.conv12(x)
        d18 = self.conv18(x)
        dcat = torch.cat((d1, d6, d12, d18), 1)
        dilatedcat = self.feat2feat(dcat)
        aspp = self.platlayer(dilatedcat)
        out = self.RCAB(aspp)

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

class ABMCA(nn.Module):
    def __init__(self, features):
        super(ABMCA, self).__init__()
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

        # aspp block
        self.asppblock = ASPP(features)
        self.compress = ChannelPool()

        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

    def forward(self, x):
            residual = x
            data = self.firstblock(x)
            aspp_data = self.compress(self.asppblock(data))
            ch_data = self.cab(data) * data
            sp_data = self.sab(aspp_data) * ch_data
            out = sp_data + residual

            return out

class SCAB(nn.Module):
    def __init__(self, features):
        super(SCAB, self).__init__()
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

        self.compress = ChannelPool()

        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

    def forward(self, x):
            residual = x
            data = self.firstblock(x)
            sp_data = self.compress(data)
            ch_data = self.cab(data) * data
            output = self.sab(sp_data) * ch_data
            out = output + residual

            return out