import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

class MHSABNReLU(nn.Module):

    def __init__(self, planes, stride, width=14, height=14, heads=4):
        super(MHSABNReLU, self).__init__()
        self.conv1 = nn.ModuleList()
        self.conv1.append(MHSA(planes, width=width, height=height, heads=heads))
        if stride == 2:
            self.conv1.append(nn.AvgPool2d(2, 2))
        self.conv1 = nn.Sequential(*self.conv1)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))

        return out

class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()

        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=2),    # stride = 1
        )
        # self.S1_MHSA = nn.Sequential(
        #          ConvBNReLU(3, 64, 3, stride=2),
        #          ConvBNReLU(64, 64, 3, stride=2),
        #          MHSABNReLU(64, stride=1, width=64, height=128, heads=4),
        #      )
        # self.S2 = nn.Sequential(
        #     ConvBNReLU(64, 64, 3, stride=2),
        #     ConvBNReLU(64, 64, 3, stride=1),
        #     # MHSABNReLU(64, stride=1, width=32, height=64, heads=4),
        #     ConvBNReLU(64, 64, 3, stride=1),    # stride = 2
        # )
        self.S2_MHSA = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            MHSABNReLU(64, stride=1, width=32, height=64, heads=4),
            # MHSABNReLU(64, stride=1, width=32, height=64, heads=4),
            ConvBNReLU(64, 64, 3, stride=1)
            )
        self.S3_MHSA = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            # ConvBNReLU(128, 128, 3, stride=1),
            MHSABNReLU(128, stride=1, width=16, height=32, heads=4),
            MHSABNReLU(128, stride=1, width=16, height=32, heads=4),
            # ConvBNReLU(128, 128, 3, stride=1)
        )
        # self.S3 = nn.Sequential(
        #     ConvBNReLU(64, 128, 3, stride=2),
        #     ConvBNReLU(128, 128, 3, stride=1),
        #     ConvBNReLU(128, 128, 3, stride=1),
        # )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2_MHSA(feat)
        feat = self.S3_MHSA(feat)
        return feat

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=2, padding=0)    # stride = 1
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=2)    # stride = 1

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

class FuseMBlayer1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(FuseMBlayer1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.se = SeModule(in_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv3[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('x =' + str(x.shape))
        feat = self.conv1(x)
        # print('conv1 =' + str(feat.shape))
        #feat = self.se(feat)
        feat = self.conv2(feat)
        # print('conv2 =' + str(feat.shape))
        feat = self.conv3(feat)
        # print('conv3 =' + str(feat.shape))
        feat = feat + x
        feat = self.relu(feat)
        return feat

class FuseMBlayer2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(FuseMBlayer2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.se = SeModule(in_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv3[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        #feat = self.se(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat

class MBLayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(MBLayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.se = SeModule(mid_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        # feat = self.se(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat

class MBLayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(MBLayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.se = SeModule(mid_chan)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv3[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.conv2(feat)
        # feat = self.se(feat)
        feat = self.conv3(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat

class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            FuseMBlayer2(16, 32),
            FuseMBlayer1(32, 32),
        )
        self.S4 = nn.Sequential(
            MBLayerS2(32, 64),
            MBLayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            MBLayerS2(64, 128),
            MBLayerS1(128, 128),
            MBLayerS1(128, 128),
            MBLayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

        # self.up1 = nn.Upsample(scale_factor=4)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        # feat5_5 = self.up1(feat5_5)
        return feat5_5

class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # self.conv1 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x_d, x_s):

        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)

        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)


        # ---------SUM----------#
        # print('out', out.size())
        # out = self.conv(x_d + x_s)

        # ---------Concate----------#
        # out = torch.cat((x_d, x_s), dim=1)
        # out = self.conv1(out)
        # out = self.conv(out)
        return out


class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes * up_factor * up_factor
        if aux:
            self.conv_out = nn.Sequential(
                ConvBNReLU(mid_chan, up_factor * up_factor, 3, stride=1),
                nn.Conv2d(up_factor * up_factor, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

# if __name__ == "__main__":

    # x = torch.randn(16, 3, 256, 512)
    # detail = DetailBranch()
    # feat = detail(x)
    # print('detail', feat.size())
    # detail torch.Size([16, 128, 16, 32])

    # x = torch.randn(16, 3, 1024, 2048)
    # stem = StemBlock()
    # feat = stem(x)
    # print('stem', feat.size())
    #
    # x = torch.randn(16, 128, 16, 32)
    # ceb = CEBlock()
    # feat = ceb(x)
    # print(feat.size())
    #
    # x = torch.randn(16, 32, 16, 32)
    # ge1 = FuseMBlayer1(32, 32)
    # feat = ge1(x)
    # print(feat.size())
    #
    # x = torch.randn(16, 16, 16, 32)
    # ge2 = FuseMBlayer2(16, 32)
    # feat = ge2(x)
    # print(feat.size())
    #
    # left = torch.randn(16, 128, 16, 32)
    # right = torch.randn(16, 128, 4, 8)
    # bga = BGALayer()
    # feat = bga(left, right)
    # print('bga', feat.size())
    # bga torch.Size([16, 128, 16, 32])

    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #

    # x = torch.randn(16, 3, 256, 512)
    # segment = SegmentBranch()
    # feat = segment(x)[0]
    # print('segment', feat.size())
    # segment torch.Size([128, 4, 8])

    # x = torch.randn(16, 3, 512, 1024)
    # model = BiSeNetV2(n_classes=19)
    # logits = model(x)[0]
    # print(logits.size())

    # model = BiSeNetV2(n_classes=2)
    # for name, param in model.named_parameters():
    #     if len(param.size()) == 1:
    #         #print(name)
    #         print(name, ':', param.size())

